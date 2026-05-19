//! CPU kernel correctness tests.

use hologram_backend::{
    Backend, BinaryCall, BufferRef, CpuBackend, KernelCall, LayoutCall, MatMulCall, ReduceCall,
    SoftmaxCall, UnaryCall, Workspace,
};

/// Minimal test workspace: a slot-indexed Vec<Vec<u8>>. The trait impls below
/// satisfy `BufferRef::offset/length` semantics by re-slicing into the slot.
struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}

impl TestWorkspace {
    fn new(slots: Vec<Vec<u8>>) -> Self {
        Self { slots }
    }
    fn slot(&self, i: u32) -> &Vec<u8> {
        &self.slots[i as usize]
    }
    fn slot_mut(&mut self, i: u32) -> &mut Vec<u8> {
        &mut self.slots[i as usize]
    }
}

impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slot(b.slot)[..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let len = self.slot(b.slot).len();
        let _ = b.length;
        &mut self.slot_mut(b.slot)[..len]
    }
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(Vec<&'a [u8]>, &'a mut [u8])> {
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

fn buf(slot: u32, length: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length,
    }
}

#[test]
fn add_byte_kernel() {
    let mut ws = TestWorkspace::new(vec![
        vec![1u8, 2, 3, 4],
        vec![10u8, 20, 30, 40],
        vec![0u8; 4],
    ]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Add(BinaryCall {
        a: buf(0, 4),
        b: buf(1, 4),
        output: buf(2, 4),
        element_count: 4,
        witt_bits: 8,
        dtype: 1,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slot(2), &vec![11u8, 22, 33, 44]);
}

#[test]
fn neg_byte_kernel() {
    let mut ws = TestWorkspace::new(vec![vec![0u8, 1, 2, 255], vec![0u8; 4]]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Neg(UnaryCall {
        input: buf(0, 4),
        output: buf(1, 4),
        element_count: 4,
        witt_bits: 8,
        dtype: 1,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slot(1), &vec![0u8, 255, 254, 1]);
}

#[test]
fn matmul_byte_2x2() {
    // A = [[1,2],[3,4]] * B = [[5,6],[7,8]] => [[19, 22],[43, 50]] mod 256
    let mut ws = TestWorkspace::new(vec![vec![1u8, 2, 3, 4], vec![5u8, 6, 7, 8], vec![0u8; 4]]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::MatMul(MatMulCall {
        a: buf(0, 4),
        b: buf(1, 4),
        output: buf(2, 4),
        m: 2,
        k: 2,
        n: 2,
        dtype: 0,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slot(2), &vec![19u8, 22, 43, 50]);
}

#[test]
fn reshape_copies_input() {
    let mut ws = TestWorkspace::new(vec![vec![1u8, 2, 3, 4, 5, 6], vec![0u8; 6]]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Reshape(LayoutCall {
        input: buf(0, 6),
        output: buf(1, 6),
        element_count: 6,
        dtype: 0,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slot(1), &vec![1u8, 2, 3, 4, 5, 6]);
}

#[test]
fn reduce_sum_byte() {
    let mut ws = TestWorkspace::new(vec![
        vec![1u8, 2, 3, 4, 5], // sum = 15
        vec![0u8; 1],
    ]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::ReduceSum(ReduceCall {
        input: buf(0, 5),
        output: buf(1, 1),
        element_count: 5,
        axis_count: 1,
        keepdims: false,
        dtype: 0,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    assert_eq!(ws.slot(1)[0], 15);
}

#[test]
fn softmax_distribution_sums_to_approx_unit() {
    // Softmax(0,0,0,0) on the byte ring should yield ~uniform distribution
    // (roughly 64 each on a 256-sum scaling).
    let mut ws = TestWorkspace::new(vec![vec![0u8, 0, 0, 0], vec![0u8; 4]]);
    let mut backend: CpuBackend<TestWorkspace> = CpuBackend::new();
    let call = KernelCall::Softmax(SoftmaxCall {
        input: buf(0, 4),
        output: buf(1, 4),
        batch: 1,
        feature: 4,
        dtype: 0,
    });
    backend.dispatch(&call, &mut ws).unwrap();
    let out = ws.slot(1);
    let total: u32 = out.iter().map(|&v| v as u32).sum();
    // Each entry should be roughly 255/4 ≈ 64; sum about 255.
    assert!((total as i32 - 255).abs() < 8, "total = {total}");
    for &v in out.iter() {
        assert!((60..=68).contains(&v), "uniform expected, got {v}");
    }
}
