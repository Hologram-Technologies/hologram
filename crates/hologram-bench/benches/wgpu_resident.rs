//! Resident vs round-trip dispatch on the canonical WebGPU backend
//! (ADR-051 step 3 verification).
//!
//! Two contrasting paths through `WgpuBackend`, both producing the
//! same canonical result:
//!
//!   * **resident** — `alloc_workspace` once, seed inputs via
//!     `write_span`, run the kernel sequence through
//!     `dispatch_resident`, then `read_span` the output. Per-call
//!     transfers are zero on migrated arms; the workspace stays on
//!     the device.
//!   * **round_trip** — keep a host `[f32]` buffer, call the legacy
//!     `dispatch(&mut [f32], call)` per kernel. Each call uploads
//!     its inputs, runs one pipeline, and reads the result back to
//!     the host slice — the pre-ADR-051 behaviour.
//!
//! Run:
//!
//! ```bash
//! cargo bench -p hologram-bench --features webgpu --bench wgpu_resident
//! ```
//!
//! The bench exits cleanly with a printed note when no wgpu adapter
//! is available so CI on headless boxes doesn't fail outright.
#![cfg(feature = "webgpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_backend::canonical::WgpuBackend;
use hologram_transform::{
    BackendWorkspace, BinaryCall, CanonicalBackend, KernelCall, MatMulCall, SlotSpan,
};

/// wgpu storage-buffer offsets must be 64-element-aligned (256 bytes
/// at f32). Round each span length up to that boundary so adjacent
/// spans don't violate `min_storage_buffer_offset_alignment`.
const ALIGN: usize = 64;

#[inline]
fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// Build the 3-op binary chain and the matching workspace layout.
///
/// Slots `a`, `b`, `c`, `d` of length `n`, each on an `ALIGN`-aligned
/// stride. The chain is:
///
///   1. `c = a + b`
///   2. `d = c * a`
///   3. `c = d - b`
///
/// All four slots stay live across calls — exactly the kind of
/// dependency pattern residency is meant to exploit.
fn binary_chain(n: usize) -> (Vec<KernelCall>, usize) {
    let stride = align_up(n);
    let a = SlotSpan { offset: 0, len: n };
    let b = SlotSpan {
        offset: stride,
        len: n,
    };
    let c = SlotSpan {
        offset: 2 * stride,
        len: n,
    };
    let d = SlotSpan {
        offset: 3 * stride,
        len: n,
    };
    let calls = vec![
        KernelCall::Add(BinaryCall { a, b, c }.into_add()),
        KernelCall::Mul(BinaryCall { a: c, b: a, c: d }),
        KernelCall::Sub(BinaryCall { a: d, b, c }),
    ];
    (calls, 4 * stride)
}

/// Adapter `BinaryCall -> AddCall` so the chain stays in `BinaryCall`
/// shape end to end without depending on `AddCall`'s field layout.
trait IntoAdd {
    fn into_add(self) -> hologram_transform::AddCall;
}
impl IntoAdd for BinaryCall {
    fn into_add(self) -> hologram_transform::AddCall {
        hologram_transform::AddCall {
            a: self.a,
            b: self.b,
            c: self.c,
        }
    }
}

fn seed_a(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 * i as f32 - 1.0).collect()
}
fn seed_b(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.05 * i as f32 + 0.25).collect()
}

fn bench_binary_chain(c: &mut Criterion, gpu: &mut WgpuBackend) {
    let mut group = c.benchmark_group("wgpu_binary_chain_3");
    // Sizes span per-call-overhead-dominated (256) up to a megabyte
    // workspace where bandwidth, not dispatch overhead, sets the cost.
    for &n in &[256_usize, 4096, 65_536, 1 << 18] {
        group.throughput(Throughput::Elements(n as u64));
        let (calls, capacity) = binary_chain(n);
        let stride = align_up(n);
        let a_data = seed_a(n);
        let b_data = seed_b(n);
        let a_span = SlotSpan { offset: 0, len: n };
        let b_span = SlotSpan {
            offset: stride,
            len: n,
        };
        let out_span = SlotSpan {
            offset: 2 * stride,
            len: n,
        };

        // Resident: workspace allocated once, reused across iters.
        let mut ws = gpu.alloc_workspace(capacity).expect("alloc workspace");
        group.bench_with_input(BenchmarkId::new("resident", n), &(), |bench, _| {
            bench.iter(|| {
                ws.write_span(a_span, black_box(&a_data)).unwrap();
                ws.write_span(b_span, black_box(&b_data)).unwrap();
                for call in &calls {
                    gpu.dispatch_resident(&mut ws, call).unwrap();
                }
                let out = ws.read_span(out_span).unwrap();
                black_box(out);
            });
        });

        // Round-trip: legacy host-slice dispatch — every call uploads
        // its inputs, runs one pipeline, downloads the result. The
        // host buffer is what every dispatch sees.
        let mut storage = vec![0.0_f32; capacity];
        group.bench_with_input(BenchmarkId::new("round_trip", n), &(), |bench, _| {
            bench.iter(|| {
                storage[..n].copy_from_slice(black_box(&a_data));
                storage[stride..stride + n].copy_from_slice(black_box(&b_data));
                for call in &calls {
                    gpu.dispatch(&mut storage, call).unwrap();
                }
                black_box(storage[2 * stride..2 * stride + n].to_vec());
            });
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion, gpu: &mut WgpuBackend) {
    let mut group = c.benchmark_group("wgpu_matmul_single");
    // Single-call shapes — matmul is the heaviest per-call kernel,
    // so this isolates the per-dispatch transfer cost rather than
    // chain residency.
    let shapes: &[(usize, usize, usize)] = &[
        (32, 32, 32),
        (128, 128, 128),
        (256, 512, 256),
        (512, 512, 512),
    ];
    for &(m, k, n) in shapes {
        let stride = align_up(m * k).max(align_up(k * n)).max(align_up(m * n));
        let capacity = 3 * stride;
        let a_data: Vec<f32> = (0..m * k).map(|i| 0.001 * i as f32 + 0.5).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| 0.001 * i as f32 - 0.25).collect();
        let a_span = SlotSpan {
            offset: 0,
            len: m * k,
        };
        let b_span = SlotSpan {
            offset: stride,
            len: k * n,
        };
        let c_span = SlotSpan {
            offset: 2 * stride,
            len: m * n,
        };
        let call = KernelCall::MatMul(MatMulCall {
            a: a_span,
            b: b_span,
            c: c_span,
            m,
            k,
            n,
        });
        group.throughput(Throughput::Elements((2 * m * k * n) as u64)); // FLOPs

        let mut ws = gpu.alloc_workspace(capacity).expect("alloc workspace");
        group.bench_with_input(
            BenchmarkId::new("resident", format!("{m}x{k}x{n}")),
            &(),
            |bench, _| {
                bench.iter(|| {
                    ws.write_span(a_span, black_box(&a_data)).unwrap();
                    ws.write_span(b_span, black_box(&b_data)).unwrap();
                    gpu.dispatch_resident(&mut ws, &call).unwrap();
                    let out = ws.read_span(c_span).unwrap();
                    black_box(out);
                });
            },
        );

        let mut storage = vec![0.0_f32; capacity];
        group.bench_with_input(
            BenchmarkId::new("round_trip", format!("{m}x{k}x{n}")),
            &(),
            |bench, _| {
                bench.iter(|| {
                    storage[..m * k].copy_from_slice(black_box(&a_data));
                    storage[stride..stride + k * n].copy_from_slice(black_box(&b_data));
                    gpu.dispatch(&mut storage, &call).unwrap();
                    black_box(storage[2 * stride..2 * stride + m * n].to_vec());
                });
            },
        );
    }
    group.finish();
}

fn bench_resident(c: &mut Criterion) {
    // Skip cleanly on hosts without a wgpu adapter (headless CI etc.)
    // rather than failing the whole bench run.
    let mut gpu = match WgpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("wgpu_resident: skipping — no wgpu adapter ({e})");
            return;
        }
    };
    bench_binary_chain(c, &mut gpu);
    bench_matmul(c, &mut gpu);
}

criterion_group!(benches, bench_resident);
criterion_main!(benches);
