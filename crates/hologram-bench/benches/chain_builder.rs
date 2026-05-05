//! Compare the three ways of building a `TransformChain`:
//!
//! 1. **procedural** — `add_tensor` for inputs *and* outputs, then
//!    `push_add(AddInputs { a, b, c })`. The original API. Outputs
//!    are caller-allocated; the builder does the minimum.
//! 2. **generic_push_op** — `add_tensor` for inputs only;
//!    `push_op(SemanticOp::Add, &[a, b])` allocates the output by
//!    inferring its shape, then emits the node.
//! 3. **fluent** — `FluentChain::input(...)` returns a `TensorRef`;
//!    chain ops via `a.add(&b)`. Same machinery as `push_op` plus a
//!    `RefCell` for shared-mutable access.
//!
//! Three chain lengths exercise the per-op overhead at different
//! amortisation horizons:
//!   * `n_ops = 8`   — bench-scale plan (≈ a single transformer block).
//!   * `n_ops = 64`  — many-block plan.
//!   * `n_ops = 512` — large/deep plan.
//!
//! Throughput is `Elements(n_ops)` so Criterion shows the per-op
//! amortised cost directly.
//!
//! All three styles build the same logical computation: a sequential
//! chain of `Add` ops over `[N]`-shaped f32 tensors, where each add's
//! output feeds the next add's left operand. No execution; we're
//! measuring build cost only.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_transform::{
    AddInputs, AddressRef, FluentChain, SemanticOp, TensorId, TransformChain,
};

/// Chain lengths covered. The widest end (512 ops) is well past any
/// real-world transformer plan, so it isolates the per-op cost from
/// fixed startup.
const CHAIN_LENGTHS: &[usize] = &[8, 64, 512];

/// Common tensor shape — small so memory traffic doesn't dominate
/// what's supposed to be a build-time benchmark.
const TENSOR_DIMS: &[usize] = &[16];

/// Build via `TransformChainBuilder::push_add` with caller-allocated
/// outputs. `reserved == true` pre-sizes the chain's vectors via
/// `builder_with_capacity` to isolate realloc cost from per-op work.
fn build_procedural(n_ops: usize, reserved: bool) -> TransformChain {
    let mut b = if reserved {
        TransformChain::builder_with_capacity(n_ops)
    } else {
        TransformChain::builder()
    };
    let initial = b.add_tensor(TENSOR_DIMS, false);
    let other = b.add_tensor(TENSOR_DIMS, false);
    let mut current = initial;
    for _ in 0..n_ops {
        let out = b.add_tensor(TENSOR_DIMS, false);
        b.push_add(AddInputs {
            a: AddressRef::of(current),
            b: AddressRef::of(other),
            c: AddressRef::of(out),
        });
        current = out;
    }
    let _final = current;
    b.build()
}

/// Build via the generic `push_op` — output tensors are auto-allocated.
fn build_generic_push_op(n_ops: usize, reserved: bool) -> TransformChain {
    let mut b = if reserved {
        TransformChain::builder_with_capacity(n_ops)
    } else {
        TransformChain::builder()
    };
    let initial = b.add_tensor(TENSOR_DIMS, false);
    let other = b.add_tensor(TENSOR_DIMS, false);
    let mut current = initial;
    for _ in 0..n_ops {
        current = b
            .push_op(SemanticOp::Add, &[current, other])
            .expect("push_op Add");
    }
    b.build()
}

/// Build via the fluent `TensorRef` API.
fn build_fluent(n_ops: usize, reserved: bool) -> TransformChain {
    let chain = if reserved {
        FluentChain::with_capacity(n_ops)
    } else {
        FluentChain::new()
    };
    let _final_id: TensorId = {
        let initial = chain.input(TENSOR_DIMS, false);
        let other = chain.input(TENSOR_DIMS, false);
        let mut current = initial;
        for _ in 0..n_ops {
            current = current.add(&other).expect("fluent add");
        }
        current.id()
    };
    chain.into_chain()
}

fn bench_chain_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_builder");
    for &n in CHAIN_LENGTHS {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("procedural", n), &n, |bench, &n| {
            bench.iter(|| black_box(build_procedural(n, false)));
        });

        group.bench_with_input(
            BenchmarkId::new("procedural_reserved", n),
            &n,
            |bench, &n| {
                bench.iter(|| black_box(build_procedural(n, true)));
            },
        );

        group.bench_with_input(BenchmarkId::new("generic_push_op", n), &n, |bench, &n| {
            bench.iter(|| black_box(build_generic_push_op(n, false)));
        });

        group.bench_with_input(
            BenchmarkId::new("generic_push_op_reserved", n),
            &n,
            |bench, &n| {
                bench.iter(|| black_box(build_generic_push_op(n, true)));
            },
        );

        group.bench_with_input(BenchmarkId::new("fluent", n), &n, |bench, &n| {
            bench.iter(|| black_box(build_fluent(n, false)));
        });

        group.bench_with_input(BenchmarkId::new("fluent_reserved", n), &n, |bench, &n| {
            bench.iter(|| black_box(build_fluent(n, true)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chain_builder);
criterion_main!(benches);
