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
    BackendWorkspace, BinaryCall, CanonicalBackend, KernelCall, MatMulCall, NormScaleCall,
    SlotSpan, SoftmaxCall,
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

/// Decode-step chain at LLaMA-shaped sizes — `matmul → add → rms_norm
/// → matmul → softmax`. This is the tail of a transformer block at a
/// single-token decode step (`m=1`). Each call takes only a few µs of
/// actual GPU compute, so the per-submit floor (~1.3 ms on Metal)
/// dominates total wall time. Comparing `dispatch_resident` (one
/// submit per call) against the new `run_resident` override (one
/// submit for the whole chain) shows how much of the per-call
/// overhead is amortisable.
fn bench_decode_step(c: &mut Criterion, gpu: &mut WgpuBackend) {
    let mut group = c.benchmark_group("wgpu_decode_step_chain");
    // (hidden, ff) — pairs roughly proportional to LLaMA-2 / Mistral
    // dimensions. The smallest pair runs in seconds; the LLaMA-2-7B
    // pair allocates ~250 MB of workspace and may not fit on small
    // GPUs.
    let shapes: &[(usize, usize)] = &[(512, 1376), (1024, 2752), (4096, 11008)];
    for &(hidden, ff) in shapes {
        // Per-tensor strides — every span lives on its own
        // 64-element-aligned offset so adjacent storage-buffer
        // bindings satisfy `min_storage_buffer_offset_alignment`.
        let h_stride = align_up(hidden);
        let ff_stride = align_up(ff);
        let attn_w_stride = align_up(hidden * hidden);
        let ff_w_stride = align_up(hidden * ff);

        let a_off = 0;
        let attn_w_off = a_off + h_stride;
        let attn_out_off = attn_w_off + attn_w_stride;
        let resid_off = attn_out_off + h_stride;
        let added_off = resid_off + h_stride;
        let normed_off = added_off + h_stride;
        let norm_w_off = normed_off + h_stride;
        let ff_w_off = norm_w_off + h_stride;
        let ff_out_off = ff_w_off + ff_w_stride;
        let softmax_out_off = ff_out_off + ff_stride;
        let capacity = softmax_out_off + ff_stride;

        let a = SlotSpan {
            offset: a_off,
            len: hidden,
        };
        let attn_w = SlotSpan {
            offset: attn_w_off,
            len: hidden * hidden,
        };
        let attn_out = SlotSpan {
            offset: attn_out_off,
            len: hidden,
        };
        let resid = SlotSpan {
            offset: resid_off,
            len: hidden,
        };
        let added = SlotSpan {
            offset: added_off,
            len: hidden,
        };
        let normed = SlotSpan {
            offset: normed_off,
            len: hidden,
        };
        let norm_w = SlotSpan {
            offset: norm_w_off,
            len: hidden,
        };
        let ff_w = SlotSpan {
            offset: ff_w_off,
            len: hidden * ff,
        };
        let ff_out = SlotSpan {
            offset: ff_out_off,
            len: ff,
        };
        let softmax_out = SlotSpan {
            offset: softmax_out_off,
            len: ff,
        };

        // Use planner-side `f32::to_bits()` for epsilon; matches the
        // rest of the wgpu backend's norm-uniform packing.
        let epsilon = 1e-5_f32.to_bits();
        let calls = vec![
            KernelCall::MatMul(MatMulCall {
                a,
                b: attn_w,
                c: attn_out,
                m: 1,
                k: hidden,
                n: hidden,
            }),
            KernelCall::Add(hologram_transform::AddCall {
                a: attn_out,
                b: resid,
                c: added,
            }),
            KernelCall::RmsNorm(NormScaleCall {
                input: added,
                weight: norm_w,
                output: normed,
                size: hidden as u32,
                epsilon,
            }),
            KernelCall::MatMul(MatMulCall {
                a: normed,
                b: ff_w,
                c: ff_out,
                m: 1,
                k: hidden,
                n: ff,
            }),
            KernelCall::Softmax(SoftmaxCall {
                input: ff_out,
                output: softmax_out,
                size: ff,
            }),
        ];

        // Throughput is dominated by the FF matmul: 2*1*hidden*ff FMAs
        // per token. Reporting that gives criterion a sensible baseline.
        group.throughput(Throughput::Elements((2 * hidden * ff) as u64));

        let a_data: Vec<f32> = (0..hidden).map(|i| 0.001 * i as f32).collect();
        let attn_w_data: Vec<f32> = (0..hidden * hidden)
            .map(|i| 0.0001 * (i % 1024) as f32 - 0.05)
            .collect();
        let resid_data: Vec<f32> = (0..hidden).map(|i| 0.002 * i as f32 - 0.1).collect();
        let norm_w_data: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.001 * i as f32).collect();
        let ff_w_data: Vec<f32> = (0..hidden * ff)
            .map(|i| 0.0001 * (i % 1024) as f32 - 0.05)
            .collect();

        let mut ws = match gpu.alloc_workspace(capacity) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "wgpu_decode_step: skipping ({hidden}/{ff}) — alloc workspace failed: {e}"
                );
                continue;
            }
        };
        ws.write_span(attn_w, &attn_w_data).unwrap();
        ws.write_span(ff_w, &ff_w_data).unwrap();
        ws.write_span(norm_w, &norm_w_data).unwrap();

        // Per-call: legacy path — each `dispatch_resident` builds and
        // submits its own command encoder, so the chain costs N submits.
        group.bench_with_input(
            BenchmarkId::new("per_call", format!("h{hidden}_ff{ff}")),
            &(),
            |bench, _| {
                bench.iter(|| {
                    ws.write_span(a, black_box(&a_data)).unwrap();
                    ws.write_span(resid, black_box(&resid_data)).unwrap();
                    for call in &calls {
                        gpu.dispatch_resident(&mut ws, call).unwrap();
                    }
                    let out = ws.read_span(softmax_out).unwrap();
                    black_box(out);
                });
            },
        );

        // Batched: new override — `run_resident` parks one encoder
        // and walks all calls into it, then submits exactly once. The
        // bind-group + uniform caches turn warm-loop iterations into
        // pure record + dispatch work.
        group.bench_with_input(
            BenchmarkId::new("batched", format!("h{hidden}_ff{ff}")),
            &(),
            |bench, _| {
                bench.iter(|| {
                    ws.write_span(a, black_box(&a_data)).unwrap();
                    ws.write_span(resid, black_box(&resid_data)).unwrap();
                    gpu.run_resident(&mut ws, &calls).unwrap();
                    let out = ws.read_span(softmax_out).unwrap();
                    black_box(out);
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
    bench_decode_step(c, &mut gpu);
}

criterion_group!(benches, bench_resident);
criterion_main!(benches);
