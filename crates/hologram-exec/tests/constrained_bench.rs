//! Micro-benchmarks for constrained execution primitives.
//!
//! Run: `cargo test -p hologram-exec --test constrained_bench --release -- --nocapture`
//!
//! Uses `std::time::Instant` — no nightly or criterion dependency required.
//! Results are printed to stdout; use `--nocapture` to see them.

use std::time::Instant;

use hologram_exec::constrained::{
    ConstrainedProfile, KernelAllowlist, KernelDiscriminant, PackedWeightSpan, RegionIndex,
    WeightWindow,
};
use hologram_graph::constant::{ConstantData, ConstantId, ConstantStore};

// ── Harness ──────────────────────────────────────────────────────────────────

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    // Warmup.
    for _ in 0..iters / 10 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed / iters;
    println!(
        "  {name:.<50} {:.1} µs/iter  ({iters} iters, {:.1} ms total)",
        per_iter.as_secs_f64() * 1e6,
        elapsed.as_secs_f64() * 1e3,
    );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_store(n: usize, size: u64) -> ConstantStore {
    let mut store = ConstantStore::new();
    for i in 0..n {
        store.insert(ConstantData::Deferred {
            byte_size: size,
            source_id: (i as u64) * size,
        });
    }
    store
}

fn make_region_index(n_constants: usize, n_regions: usize) -> RegionIndex {
    let mut idx = RegionIndex::new();
    for i in 0..n_constants {
        idx.insert(
            ConstantId::new(i as u32),
            PackedWeightSpan {
                offset: (i * 1024) as u64,
                len: 1024,
                region_id: (i % n_regions) as u32,
            },
        );
    }
    idx
}

fn make_tape_1000() -> hologram_exec::tape::EnumTape {
    use hologram_exec::tape::{EnumTape, TapeInstruction, TapeKernel};

    let mut tape = EnumTape::new();
    for i in 0..1000u32 {
        tape.instructions.push(TapeInstruction {
            kernel: match i % 3 {
                0 => TapeKernel::InlineAdd,
                1 => TapeKernel::InlineMul,
                _ => TapeKernel::InlineRelu,
            },
            input_indices: smallvec::smallvec![],
            output_idx: i,
            output_byte_hint: 0,
            output_elem_size: 4,
            weight_offset_hint: 0,
            passthrough: false,
            can_reuse_input: false,
            output_meta: None,
            fast_path: Default::default(),
            shape_source: Default::default(),
        });
    }
    tape
}

// ── Tests (benchmarks) ───────────────────────────────────────────────────────

#[test]
fn bench_weight_window() {
    println!("\n=== Weight Window Benchmarks ===");

    let store64 = make_store(64, 4096);
    bench("ensure_sequential_64x4KB_window16", 1000, || {
        let mut ww = WeightWindow::new(16 * 4096);
        for i in 0..64u32 {
            ww.ensure(&[ConstantId::new(i)], &store64).unwrap();
        }
        std::hint::black_box(ww.current_usage());
    });

    let store128 = make_store(128, 8192);
    bench("ensure_evict_cycle_128x8KB_window4", 500, || {
        let mut ww = WeightWindow::new(4 * 8192);
        for i in 0..128u32 {
            ww.ensure(&[ConstantId::new(i)], &store128).unwrap();
        }
        std::hint::black_box(ww.current_usage());
    });

    let store8 = make_store(8, 1024);
    bench("dedup_no_evict_8x1KB_100_rounds", 500, || {
        let mut ww = WeightWindow::new(1024 * 1024);
        for _ in 0..100 {
            for i in 0..8u32 {
                ww.ensure(&[ConstantId::new(i)], &store8).unwrap();
            }
        }
        std::hint::black_box(ww.current_usage());
    });
}

#[test]
fn bench_region_index() {
    println!("\n=== Region Index Benchmarks ===");

    let idx100 = make_region_index(100, 10);
    bench("lookup_100_constants", 5000, || {
        let mut sum = 0u64;
        for i in 0..100u32 {
            if let Some(span) = idx100.get(ConstantId::new(i)) {
                sum += span.offset;
            }
        }
        std::hint::black_box(sum);
    });

    let idx1000 = make_region_index(1000, 10);
    bench("constants_in_region_1000x10", 500, || {
        let mut total = 0;
        for r in 0..10u32 {
            total += idx1000.constants_in_region(r).len();
        }
        std::hint::black_box(total);
    });

    bench("byte_range_1000x10", 5000, || {
        let mut total = 0u64;
        for r in 0..10u32 {
            if let Some((start, end)) = idx1000.region_byte_range(r) {
                total += end - start;
            }
        }
        std::hint::black_box(total);
    });
}

#[test]
fn bench_discriminant_and_validation() {
    println!("\n=== Kernel Discriminant & Tape Validation Benchmarks ===");

    use hologram_exec::tape::TapeKernel;

    let kernels = vec![
        TapeKernel::InlineAdd,
        TapeKernel::InlineMul,
        TapeKernel::InlineRmsNorm {
            size: 128,
            epsilon: f32::to_bits(1e-5),
        },
        TapeKernel::InlineSoftmax { size: 64 },
        TapeKernel::Output,
        TapeKernel::InlineMatMul {
            m: 1,
            k: 128,
            n: 128,
        },
    ];

    bench("discriminant_extraction_600", 10000, || {
        for _ in 0..100 {
            for k in &kernels {
                std::hint::black_box(KernelDiscriminant::from_kernel(k));
            }
        }
    });

    let tape = make_tape_1000();

    let profile = ConstrainedProfile {
        kernel_allowlist: Some(KernelAllowlist::compute()),
        ..Default::default()
    };

    use hologram_exec::constrained::validate_constrained_tape;

    bench("validate_tape_1000_instructions", 5000, || {
        validate_constrained_tape(&tape, &profile).unwrap();
        std::hint::black_box(());
    });

    let profile_no_allowlist = ConstrainedProfile::default();
    bench("validate_tape_1000_no_allowlist", 5000, || {
        validate_constrained_tape(&tape, &profile_no_allowlist).unwrap();
        std::hint::black_box(());
    });
}
