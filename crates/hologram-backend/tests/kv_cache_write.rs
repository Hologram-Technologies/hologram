//! Witnesses for the `KvCacheWrite` kernel — the honest-copy reference form
//! of the fixed-bucket KV-cache row write. The executor's in-place-move
//! elision (hologram-exec) is pinned bit-identical to *this* kernel by the
//! end-to-end tests there; this file pins the kernel itself against a
//! straight-line oracle, including the ring-wrap case and every refusal.

use hologram_backend::{
    Backend, BufferRef, CpuBackend, KernelCall, KvCacheWriteCall, SplitReads, Workspace,
};

/// Slot-indexed test workspace (the established direct-dispatch pattern).
struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}
impl TestWorkspace {
    fn push(&mut self, data: &[u8]) -> BufferRef {
        let slot = self.slots.len() as u32;
        self.slots.push(data.to_vec());
        BufferRef {
            slot,
            offset: 0,
            length: data.len() as u64,
        }
    }
}
impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let len = self.slots[b.slot as usize].len();
        let _ = b.length;
        &mut self.slots[b.slot as usize][..len]
    }
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])> {
        let w = write.slot as usize;
        if reads.iter().any(|r| r.slot as usize == w) {
            return None;
        }
        let (lo, hi) = self.slots.split_at_mut(w);
        let (wbuf, hi_rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &hi_rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

/// Straight-line oracle: copy the cache, then overwrite the wrapped rows.
fn oracle(
    cache: &[u8],
    new: &[u8],
    pos: u32,
    planes: usize,
    bucket: usize,
    rows: usize,
    row: usize,
) -> Vec<u8> {
    let mut out = cache.to_vec();
    let pos = pos as usize % bucket;
    for p in 0..planes {
        for j in 0..rows {
            let dst = p * bucket * row + ((pos + j) % bucket) * row;
            let src = p * rows * row + j * row;
            out[dst..dst + row].copy_from_slice(&new[src..src + row]);
        }
    }
    out
}

fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn run(planes: u32, bucket: u32, rows: u32, row: u32, pos: u32) -> (Vec<u8>, Vec<u8>) {
    let cache = bytes((planes * bucket * row) as usize, 3);
    let new = bytes((planes * rows * row) as usize, 7);
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rc = ws.push(&cache);
    let rn = ws.push(&new);
    let rp = ws.push(&pos.to_le_bytes());
    let ro = ws.push(&vec![0u8; cache.len()]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::KvCacheWrite(KvCacheWriteCall {
            cache: rc,
            new: rn,
            pos: rp,
            output: ro,
            planes,
            bucket_rows: bucket,
            new_rows: rows,
            row_bytes: row,
        }),
        &mut ws,
    )
    .unwrap();
    let want = oracle(
        &cache,
        &new,
        pos,
        planes as usize,
        bucket as usize,
        rows as usize,
        row as usize,
    );
    (ws.slots[ro.slot as usize].clone(), want)
}

/// The plain decode step: one row appended mid-bucket, every plane.
#[test]
fn single_row_write_matches_the_oracle_bytewise() {
    let (got, want) = run(2, 8, 1, 16, 5);
    assert_eq!(got, want);
}

/// Chunked append that wraps the ring: rows land at `bucket-1` and `0`.
#[test]
fn wrapping_write_matches_the_oracle_bytewise() {
    let (got, want) = run(3, 8, 4, 12, 6);
    assert_eq!(got, want);
}

/// A position at (and beyond) the bucket wraps — ring semantics, any u32.
#[test]
fn position_wraps_modulo_the_bucket() {
    let (got, want) = run(1, 8, 2, 4, 8);
    assert_eq!(got, want);
    let (got, want) = run(1, 8, 2, 4, 8 * 1000 + 3);
    assert_eq!(got, want);
}

/// `new_rows == bucket_rows` replaces the whole bucket (rotated by pos).
#[test]
fn full_bucket_replacement_matches_the_oracle() {
    let (got, want) = run(2, 6, 6, 8, 4);
    assert_eq!(got, want);
}

/// Refusals: geometry the kernel cannot honor fails loud, never truncates.
#[test]
fn invalid_geometry_and_short_operands_are_refused() {
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    let build =
        |ws: &mut TestWorkspace, cache_n: usize, new_n: usize, pos_n: usize, out_n: usize| {
            let rc = ws.push(&bytes(cache_n, 1));
            let rn = ws.push(&bytes(new_n, 2));
            let rp = ws.push(&bytes(pos_n, 0));
            let ro = ws.push(&vec![0u8; out_n]);
            (rc, rn, rp, ro)
        };
    let call = |rc, rn, rp, ro, planes, bucket, rows, row| {
        KernelCall::KvCacheWrite(KvCacheWriteCall {
            cache: rc,
            new: rn,
            pos: rp,
            output: ro,
            planes,
            bucket_rows: bucket,
            new_rows: rows,
            row_bytes: row,
        })
    };
    // new_rows > bucket_rows: later rows would silently clobber earlier ones.
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 2 * 4, 3 * 4, 4, 2 * 4);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 3, 4), &mut ws)
        .is_err());
    // Zero bucket / zero row width.
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 8, 4, 4, 8);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 0, 1, 4), &mut ws)
        .is_err());
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 8, 4, 4, 8);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 1, 0), &mut ws)
        .is_err());
    // Short cache / short new / short pos / short output.
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 7, 4, 4, 8);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 1, 4), &mut ws)
        .is_err());
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 8, 3, 4, 8);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 1, 4), &mut ws)
        .is_err());
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 8, 4, 3, 8);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 1, 4), &mut ws)
        .is_err());
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rc, rn, rp, ro) = build(&mut ws, 8, 4, 4, 7);
    assert!(be
        .dispatch(&call(rc, rn, rp, ro, 1, 2, 1, 4), &mut ws)
        .is_err());
}
