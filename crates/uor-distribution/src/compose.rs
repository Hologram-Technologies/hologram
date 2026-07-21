//! Composition canonical byte forms (spec §5.4 / Appendix B) for the five categorical operations
//! `g2`/`f4`/`e6`/`e7`/`e8`, plus the self-describing witness header (§3.6).
//!
//! The composed κ-label is the hash of [`compose_canonical`] under the operands' **shared** σ-axis
//! (§5.4 homogeneity). This module owns the byte layout — the same cross-registry contract role as
//! [`crate::edge`]: two registries that canonicalize a composition differently mint different composed
//! κ-labels. Operand κ-labels are passed as their on-the-wire bytes (`axis:hexdigest`).

use alloc::vec::Vec;

/// A categorical composition operation (spec §5.4). `g2` is a commutative binary product; the rest
/// are unary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Commutative binary product — lex-min-first concatenation (B.1).
    G2,
    /// Involution quotient — lex-min of the digest and its bitwise complement (B.2).
    F4,
    /// Degree partition — a tag byte from `first_byte % 9` (B.3).
    E6,
    /// S₄ orbit — lex-min over the 24 quarter-permutations of the digest (B.4).
    E7,
    /// Identity embedding — the operand's canonical bytes (B.5).
    E8,
}

impl Op {
    /// Parse a `POST /v2/{path}/compose/{op}` operation token.
    pub fn parse(s: &str) -> Option<Op> {
        match s {
            "g2" => Some(Op::G2),
            "f4" => Some(Op::F4),
            "e6" => Some(Op::E6),
            "e7" => Some(Op::E7),
            "e8" => Some(Op::E8),
            _ => None,
        }
    }
    /// The operation token.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::G2 => "g2",
            Op::F4 => "f4",
            Op::E6 => "e6",
            Op::E7 => "e7",
            Op::E8 => "e8",
        }
    }
    /// The number of operands the operation takes (`g2` is binary; the rest are unary).
    pub fn arity(self) -> usize {
        if matches!(self, Op::G2) {
            2
        } else {
            1
        }
    }
}

/// Why a composition's canonical form could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeError {
    /// The operand count does not match the operation's arity.
    BadArity,
    /// Operands do not all share one σ-axis (§5.4 homogeneity → `AXIS_MISMATCH`).
    AxisMismatch,
    /// An operand is not a well-formed `axis:hexdigest` κ-label.
    MalformedOperand,
}

/// The canonical byte form of a composition (spec Appendix B). All operands MUST share a σ-axis.
/// The composed κ is `hash(compose_canonical(..))` under that axis.
pub fn compose_canonical(op: Op, operands: &[&[u8]]) -> Result<Vec<u8>, ComposeError> {
    if operands.len() != op.arity() {
        return Err(ComposeError::BadArity);
    }
    let axis0 = axis_of(operands[0]).ok_or(ComposeError::MalformedOperand)?;
    for o in operands.iter() {
        if axis_of(o) != Some(axis0) {
            return Err(ComposeError::AxisMismatch);
        }
    }
    match op {
        Op::G2 => Ok(g2(operands[0], operands[1])),
        Op::F4 => f4(operands[0]),
        Op::E6 => e6(operands[0]),
        Op::E7 => e7(operands[0]),
        Op::E8 => Ok(operands[0].to_vec()), // identity embedding (B.5)
    }
}

/// A §3.6 self-describing witness blob: `[label_width u16le][fingerprint_width u16le][trace_count
/// u16le][trace]`. The header lets a consumer determine the witness's parametric type without external
/// metadata; replaying the trace (re-hashing it under the σ-axis) re-derives the κ without the
/// original content.
pub fn witness_blob(label_width: u16, fingerprint_width: u16, trace: &[u8]) -> Vec<u8> {
    let mut w = Vec::with_capacity(6 + trace.len());
    w.extend_from_slice(&label_width.to_le_bytes());
    w.extend_from_slice(&fingerprint_width.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // one derivation event
    w.extend_from_slice(trace);
    w
}

/// The trace bytes of a §3.6 witness blob (everything after the 6-byte header).
pub fn witness_trace(witness: &[u8]) -> Option<&[u8]> {
    witness.get(6..)
}

// ─────────────────────────────── the five canonical forms ───────────────────────────────

fn axis_of(label: &[u8]) -> Option<&str> {
    let colon = label.iter().position(|&b| b == b':')?;
    core::str::from_utf8(&label[..colon]).ok()
}

fn digest_of(label: &[u8]) -> Option<Vec<u8>> {
    let colon = label.iter().position(|&b| b == b':')?;
    let hex = &label[colon + 1..];
    if hex.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut pairs = hex.chunks_exact(2);
    for pair in pairs.by_ref() {
        out.push((hexval(pair[0])? << 4) | hexval(pair[1])?);
    }
    if !pairs.remainder().is_empty() {
        return None; // odd-length hex digest
    }
    Some(out)
}

fn hexval(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

/// B.1 — commutative binary product: lex-min of the two concatenation orders.
fn g2(a: &[u8], b: &[u8]) -> Vec<u8> {
    let ab = [a, b].concat();
    let ba = [b, a].concat();
    if ab <= ba {
        ab
    } else {
        ba
    }
}

/// B.2 — involution quotient: axis ‖ lex-min(digest, ~digest).
fn f4(a: &[u8]) -> Result<Vec<u8>, ComposeError> {
    let axis = axis_of(a).ok_or(ComposeError::MalformedOperand)?;
    let digest = digest_of(a).ok_or(ComposeError::MalformedOperand)?;
    let complement: Vec<u8> = digest.iter().map(|b| !b).collect();
    let m = core::cmp::min(digest, complement);
    Ok(prefixed(axis, &m))
}

/// B.3 — degree partition: `[tag] ‖ operand`, tag = 0x05 if `first_byte % 9 ∈ [0,7]` else 0x06.
fn e6(a: &[u8]) -> Result<Vec<u8>, ComposeError> {
    let digest = digest_of(a).ok_or(ComposeError::MalformedOperand)?;
    let first = *digest.first().ok_or(ComposeError::MalformedOperand)?;
    let tag: u8 = if first % 9 <= 7 { 0x05 } else { 0x06 };
    let mut out = Vec::with_capacity(1 + a.len());
    out.push(tag);
    out.extend_from_slice(a);
    Ok(out)
}

/// B.4 — S₄ orbit: axis ‖ lex-min over the 24 quarter-permutations of the digest.
fn e7(a: &[u8]) -> Result<Vec<u8>, ComposeError> {
    let axis = axis_of(a).ok_or(ComposeError::MalformedOperand)?;
    let digest = digest_of(a).ok_or(ComposeError::MalformedOperand)?;
    if digest.is_empty() || digest.len() & 3 != 0 {
        return Err(ComposeError::MalformedOperand); // digest must divide into 4 quarters
    }
    let q = digest.len() / 4;
    let quarter = |i: usize| &digest[i * q..(i + 1) * q];
    let mut best: Option<Vec<u8>> = None;
    for perm in PERMS4 {
        let mut cand = Vec::with_capacity(digest.len());
        for &idx in &perm {
            cand.extend_from_slice(quarter(idx));
        }
        if best.as_ref().is_none_or(|b| cand < *b) {
            best = Some(cand);
        }
    }
    Ok(prefixed(axis, &best.unwrap()))
}

fn prefixed(axis: &str, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(axis.len() + 1 + body.len());
    out.extend_from_slice(axis.as_bytes());
    out.push(b':');
    out.extend_from_slice(body);
    out
}

/// The 24 permutations of the four quarters (S₄).
const PERMS4: [[usize; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::new();
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
    fn label(axis: &str, digest: &[u8]) -> Vec<u8> {
        prefixed(axis, hex(digest).as_bytes())
    }

    /// KD-14 — the five composition canonical forms match spec Appendix B: g2 is commutative, f4 is
    /// involution-invariant, e6 tags by degree, e7 is quarter-permutation-invariant, e8 is identity;
    /// σ-axis homogeneity + arity are enforced; the §3.6 witness header round-trips.
    #[test]
    fn kd14_composition_canonical_forms_and_witness_header() {
        let da: [u8; 32] = [0x11; 32];
        let db: [u8; 32] = [0x22; 32];
        let a = label("blake3", &da);
        let b = label("blake3", &db);

        // g2 is commutative (B.1): g2(a,b) == g2(b,a), and equals the lex-min concatenation.
        let ab = compose_canonical(Op::G2, &[&a, &b]).unwrap();
        assert_eq!(ab, compose_canonical(Op::G2, &[&b, &a]).unwrap());
        let expected = if [a.as_slice(), b.as_slice()].concat() <= [b.as_slice(), a.as_slice()].concat()
        {
            [a.as_slice(), b.as_slice()].concat()
        } else {
            [b.as_slice(), a.as_slice()].concat()
        };
        assert_eq!(ab, expected);

        // f4 is involution-invariant (B.2): f4(A) == f4(mirror(A)) where mirror flips every bit.
        let mirror_digest: Vec<u8> = da.iter().map(|b| !b).collect();
        let a_mirror = label("blake3", &mirror_digest);
        assert_eq!(
            compose_canonical(Op::F4, &[&a]).unwrap(),
            compose_canonical(Op::F4, &[&a_mirror]).unwrap()
        );

        // e6 prepends a degree tag byte (B.3): first byte 0x11 = 17, 17 % 9 = 8 ∉ [0,7] → 0x06.
        let e6 = compose_canonical(Op::E6, &[&a]).unwrap();
        assert_eq!(e6[0], 0x06);
        assert_eq!(&e6[1..], a.as_slice());

        // e7 is quarter-permutation-invariant (B.4): permuting the operand's quarters is a no-op.
        let mut swapped = Vec::new();
        swapped.extend_from_slice(&da[8..16]);
        swapped.extend_from_slice(&da[0..8]);
        swapped.extend_from_slice(&da[16..24]);
        swapped.extend_from_slice(&da[24..32]);
        let a_perm = label("blake3", &swapped);
        assert_eq!(
            compose_canonical(Op::E7, &[&a]).unwrap(),
            compose_canonical(Op::E7, &[&a_perm]).unwrap()
        );

        // e8 is identity (B.5).
        assert_eq!(compose_canonical(Op::E8, &[&a]).unwrap(), a);

        // σ-axis homogeneity (§5.4): mixed axes → AxisMismatch.
        let sa = label("sha256", &da);
        assert_eq!(
            compose_canonical(Op::G2, &[&a, &sa]),
            Err(ComposeError::AxisMismatch)
        );
        // Arity: g2 needs 2, e8 needs 1.
        assert_eq!(compose_canonical(Op::G2, &[&a]), Err(ComposeError::BadArity));
        assert_eq!(
            compose_canonical(Op::E8, &[&a, &b]),
            Err(ComposeError::BadArity)
        );

        // Witness §3.6 self-describing header + trace round-trip.
        let w = witness_blob(71, 32, &ab);
        assert_eq!(&w[0..2], &71u16.to_le_bytes());
        assert_eq!(&w[2..4], &32u16.to_le_bytes());
        assert_eq!(witness_trace(&w), Some(ab.as_slice()));
    }
}
