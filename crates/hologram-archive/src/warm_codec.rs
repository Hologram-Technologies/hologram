//! Warm-start section codec (WS class) — the constant-derivation lattice.
//!
//! A κ-label is a deterministic function of the compiled graph
//! ([`derive_label`]: op signature ‖ ordered operand labels). The op
//! signature/params are fixed at compile time and a constant/weight leaf
//! addresses to a fixed label by its content, so **every node whose
//! transitive inputs are all constants** — the *constant-only cone* (weight
//! preprocessing, dequant, bias/transpose folds) — has a determined label
//! *and* result. Warm-start makes the runtime cache **never cold**:
//!
//! * **Lattice — derived at load, not baked.** The labels are recomputable
//!   from the compiled graph, so the runtime derives the cone lattice itself
//!   when a session loads ([`derive_cone_lattice`], post-fusion so it matches
//!   what the walk dispatches). No redundant copy is stored in the archive.
//!   These labels are the reuse keys the walk mints, the keys a materialized
//!   fold pins under, and the keys a persisted κ-store resolves.
//! * **Fold — results in the archive.** `hologram_exec::fold_archive` runs
//!   the cone through the real runtime and writes this section with each
//!   cone node's materialized `result`; on load the session pins them under
//!   `label`, and the **existing** residency check in the node walk elides
//!   the cone on the first run — no walk changes, no second code path.
//!
//! This codec encodes/decodes that section, and [`derive_cone_lattice`] is
//! the shared (labels-only) derivation the runtime uses. Wire form:
//! `[u32 count] (entry)*`, entry = `[u32 slot][71 B label][u32 result_len][result_len B]`.

use crate::address::{derive_label, ContentLabel};
use crate::error::ArchiveError;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};
use hologram_backend::{buffers, KernelCall};
use smallvec::SmallVec;

/// One constant-only-cone node: its output slot, its lattice κ-label (the
/// cheap [`derive_label`] reuse key), and — once materialized by
/// `fold_archive` — its result bytes (empty for the labels-only lattice the
/// runtime derives at load).
#[derive(Debug, Clone)]
pub struct WarmEntry {
    pub slot: u32,
    pub label: ContentLabel,
    /// Materialized result of this cone node. Empty ⇒ labels-only (the
    /// lattice the runtime derives at load); filled by `fold_archive`.
    pub result: Vec<u8>,
}

/// Derive the constant-only-cone lattice for a kernel-call sequence.
///
/// Forward dataflow over `calls` in schedule order: a node is in the cone
/// iff every one of its operand slots is a constant leaf or an already-known
/// cone node (and none is a graph input). Its κ-label is the cheap
/// [`derive_label`] of its op signature with its operands' labels — exactly
/// what the runtime walk derives for that node — so `baked == derived`.
///
/// `constant_slot_labels` are the `(slot, leaf-label)` of every model
/// constant; `input_slots` the graph-input slots. The returned entries have
/// an empty `result` (labels only); `fold_archive` fills `result` after
/// materializing the cone. The runtime calls this at load to key the
/// persisted κ-store and to drive the materialized fold.
#[must_use]
pub fn derive_cone_lattice(
    calls: &[KernelCall],
    constant_slot_labels: &[(u32, ContentLabel)],
    input_slots: &[u32],
) -> Vec<WarmEntry> {
    let mut known: HashMap<u32, ContentLabel> = constant_slot_labels.iter().copied().collect();
    let inputs: HashSet<u32> = input_slots.iter().copied().collect();
    let mut out: Vec<WarmEntry> = Vec::new();

    for call in calls {
        let refs = buffers(call);
        let (output, ins) = match refs.split_last() {
            Some(v) => v,
            None => continue,
        };
        let out_slot = output.slot;
        if out_slot == u32::MAX {
            continue;
        }
        // Gather operand labels in deterministic order; the node is in the
        // cone only if every operand is constant-derivable.
        let mut in_labels: SmallVec<[ContentLabel; 4]> = SmallVec::new();
        let mut in_cone = true;
        for r in ins {
            if r.slot == u32::MAX {
                continue;
            }
            if inputs.contains(&r.slot) {
                in_cone = false;
                break;
            }
            match known.get(&r.slot) {
                Some(l) => in_labels.push(*l),
                None => {
                    in_cone = false;
                    break;
                }
            }
        }
        if !in_cone {
            continue;
        }
        let sig = call.op_signature();
        let label = derive_label(sig.opcode, sig.params(), &in_labels);
        known.insert(out_slot, label);
        out.push(WarmEntry {
            slot: out_slot,
            label,
            result: Vec::new(),
        });
    }
    out
}

pub fn encode(entries: &[WarmEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        4 + entries
            .iter()
            .map(|e| 4 + 71 + 4 + e.result.len())
            .sum::<usize>(),
    );
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.slot.to_le_bytes());
        out.extend_from_slice(e.label.as_bytes());
        out.extend_from_slice(&(e.result.len() as u32).to_le_bytes());
        out.extend_from_slice(&e.result);
    }
    out
}

pub fn decode(bytes: &[u8]) -> Result<Vec<WarmEntry>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut cur = 4usize;
    for _ in 0..count {
        let head = cur + 4 + 71 + 4;
        if head > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: head,
                actual: bytes.len(),
            });
        }
        let slot = u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let label = ContentLabel::from_bytes(&bytes[cur..cur + 71])
            .map_err(|_| ArchiveError::Io("malformed warm-start κ-label"))?;
        cur += 71;
        let len = u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap()) as usize;
        cur += 4;
        if cur + len > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cur + len,
                actual: bytes.len(),
            });
        }
        let result = bytes[cur..cur + len].to_vec();
        cur += len;
        out.push(WarmEntry {
            slot,
            label,
            result,
        });
    }
    Ok(out)
}
