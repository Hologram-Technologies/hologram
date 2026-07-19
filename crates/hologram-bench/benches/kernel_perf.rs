//! Vectorized-kernel throughput benchmark.
//!
//! Drives the f32 fast paths that the vectorization sweep added — norm /
//! softmax / activation / RoPE / elementwise — through the real
//! `CpuBackend::dispatch` production path at decode/prefill-representative
//! shapes. Each kernel runs on a `BufferArena` workspace so the split-borrow +
//! `bytemuck::cast_slice` zero-copy path is exercised exactly as in production.
//!
//! To measure the before/after of the sweep, run this bench with the three CPU
//! kernel files (`cpu/simd.rs`, `cpu/float_kernels.rs`, `cpu/kernels.rs`)
//! checked out from `main` vs from the branch — the dispatch API is identical,
//! so the same bench binary times both.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compute::cpu::dtype::DTYPE_F32;
use hologram_compute::{
    Backend, BinaryCall, BufferRef, CpuBackend, KernelCall, NormCall, RoPECall, SoftmaxCall,
    UnaryCall,
};
use hologram_exec::{BufferArena, SlotSpan};

/// `(slot index, value generator)` — one seeding rule per input slot.
type Seed = (usize, fn(usize) -> f32);

fn ref_buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// Build a `BufferArena` with one 64-byte-aligned slot per entry in
/// `slot_elems` (element counts; each slot is sized `elems * 4` bytes, rounded
/// up to 64 so the per-slot offset stays aligned for `cast_slice::<u8, f32>`).
fn make_arena(slot_elems: &[usize]) -> BufferArena {
    let mut offset = 0u64;
    let mut slots = Vec::with_capacity(slot_elems.len());
    for &e in slot_elems {
        let rounded = (e * 4).next_multiple_of(64) as u64;
        slots.push(SlotSpan {
            offset,
            length: rounded,
        });
        offset += rounded;
    }
    BufferArena::with_capacity(offset as usize, slots)
}

/// Seed `slot` with `n` deterministic f32 values (LE-encoded).
fn seed(ws: &mut BufferArena, slot: usize, n: usize, f: impl Fn(usize) -> f32) {
    let bytes = ws.write_slot(slot).expect("slot in range");
    for i in 0..n {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&f(i).to_le_bytes());
    }
}

fn run(c: &mut Criterion, name: &str, slots: &[usize], seeds: &[Seed], call: KernelCall) {
    let mut ws = make_arena(slots);
    for &(slot, f) in seeds {
        seed(&mut ws, slot, slots[slot], f);
    }
    let mut backend: CpuBackend<BufferArena> = CpuBackend::new();
    c.bench_function(name, |b| {
        b.iter(|| {
            backend.dispatch(black_box(&call), &mut ws).unwrap();
        });
    });
}

// Representative decode/prefill shapes (hidden = 4096, head_dim = 128).
const HID: usize = 4096;
const BATCH: usize = 32; // tokens / rows folded per dispatch

fn bench_rms_norm(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "rms_norm f32 32x4096",
        &[n, HID, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01 - 0.5),
            (1, |i| 1.0 + (i % 13) as f32 * 0.01),
        ],
        KernelCall::RmsNorm(NormCall {
            x: ref_buf(0),
            gamma: ref_buf(1),
            beta: NormCall::NO_RESIDUAL,
            residual: NormCall::NO_RESIDUAL,
            output: ref_buf(2),
            batch: BATCH as u32,
            feature: HID as u32,
            channels: 0,
            num_groups: 0,
            epsilon_bits: 0,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_layer_norm(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "layer_norm f32 32x4096",
        &[n, HID, HID, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01 - 0.5),
            (1, |i| 1.0 + (i % 13) as f32 * 0.01),
            (2, |i| (i % 7) as f32 * 0.01),
        ],
        KernelCall::LayerNorm(NormCall {
            x: ref_buf(0),
            gamma: ref_buf(1),
            beta: ref_buf(2),
            residual: NormCall::NO_RESIDUAL,
            output: ref_buf(3),
            batch: BATCH as u32,
            feature: HID as u32,
            channels: 0,
            num_groups: 0,
            epsilon_bits: 0,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_group_norm(c: &mut Criterion) {
    // Vision-ish: 8 samples × 32 channels × 1024 spatial, 32 groups (InstanceNorm).
    let batch = 8usize;
    let ch = 32usize;
    let spatial = 1024usize;
    let feat = ch * spatial;
    let n = batch * feat;
    run(
        c,
        "group_norm f32 8x32x1024",
        &[n, ch, ch, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01 - 0.5),
            (1, |i| 1.0 + (i % 13) as f32 * 0.01),
            (2, |i| (i % 7) as f32 * 0.01),
        ],
        KernelCall::GroupNorm(NormCall {
            x: ref_buf(0),
            gamma: ref_buf(1),
            beta: ref_buf(2),
            residual: NormCall::NO_RESIDUAL,
            output: ref_buf(3),
            batch: batch as u32,
            feature: feat as u32,
            channels: ch as u32,
            num_groups: ch as u32,
            epsilon_bits: 0,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_softmax(c: &mut Criterion) {
    // 64 attention rows × 2048 keys.
    let batch = 64usize;
    let feat = 2048usize;
    let n = batch * feat;
    run(
        c,
        "softmax f32 64x2048",
        &[n, n],
        &[(0, |i| (i % 211) as f32 * 0.02 - 2.0)],
        KernelCall::Softmax(SoftmaxCall {
            input: ref_buf(0),
            output: ref_buf(1),
            batch: batch as u32,
            feature: feat as u32,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_silu(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "silu f32 32x4096",
        &[n, n],
        &[(0, |i| (i % 211) as f32 * 0.05 - 5.0)],
        KernelCall::Silu(UnaryCall {
            input: ref_buf(0),
            output: ref_buf(1),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_gelu(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "gelu f32 32x4096",
        &[n, n],
        &[(0, |i| (i % 211) as f32 * 0.05 - 5.0)],
        KernelCall::Gelu(UnaryCall {
            input: ref_buf(0),
            output: ref_buf(1),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_rope(c: &mut Criterion) {
    // 64 heads × 128 head_dim.
    let head_dim = 128usize;
    let heads = 64usize;
    let n = heads * head_dim;
    run(
        c,
        "rope f32 64x128",
        &[n, n, n, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01 - 0.5),
            (1, |i| ((i % 128) as f32 * 0.02).cos()),
            (2, |i| ((i % 128) as f32 * 0.02).sin()),
        ],
        KernelCall::RotaryEmbedding(RoPECall {
            x: ref_buf(0),
            cos: ref_buf(1),
            sin: ref_buf(2),
            output: ref_buf(3),
            head_dim: head_dim as u32,
            element_count: n as u64,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_add(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "add f32 32x4096",
        &[n, n, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01),
            (1, |i| (i % 89) as f32 * 0.02),
        ],
        KernelCall::Add(BinaryCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
    );
}

fn bench_mul(c: &mut Criterion) {
    let n = BATCH * HID;
    run(
        c,
        "mul f32 32x4096",
        &[n, n, n],
        &[
            (0, |i| (i % 97) as f32 * 0.01),
            (1, |i| (i % 89) as f32 * 0.02),
        ],
        KernelCall::Mul(BinaryCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
    );
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_layer_norm,
    bench_group_norm,
    bench_softmax,
    bench_silu,
    bench_gelu,
    bench_rope,
    bench_add,
    bench_mul
);
criterion_main!(benches);
