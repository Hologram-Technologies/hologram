//! Source parser/lowering benchmarks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compiler::source::{self, SourceLanguage};
use hologram_compiler::{compile_from_source_language, BackendKind};
use prism::vocabulary::WittLevel;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[derive(Clone, Copy)]
struct AllocSnapshot {
    allocations: usize,
    bytes: usize,
}

fn bench_source_lowering(c: &mut Criterion) {
    let linear = SourceCase::linear_chain(512);
    let constant = SourceCase::large_constant(4096);
    bench_case(c, &linear);
    bench_case(c, &constant);
}

fn bench_case(c: &mut Criterion, case: &SourceCase) {
    case.assert_archive_equivalent();
    bench_parse_allocations(c, case);
    bench_lowering(c, case);
}

fn bench_parse_allocations(c: &mut Criterion, case: &SourceCase) {
    let name = format!("source_lowering::{}_parse_allocations", case.name);
    c.bench_function(&name, |b| {
        b.iter(|| black_box(measure_parse_allocations(black_box(&case.source))))
    });
}

fn bench_lowering(c: &mut Criterion, case: &SourceCase) {
    let program = parse_source(&case.source);
    let name = format!("source_lowering::{}_lower_ir", case.name);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let graph = source::lower_ir(black_box(&program)).unwrap();
            black_box(graph.node_count());
        })
    });
}

fn measure_parse_allocations(source: &str) -> AllocSnapshot {
    reset_allocations();
    let program = parse_source(source);
    let snapshot = allocation_snapshot();
    black_box((program.items().len(), snapshot.allocations, snapshot.bytes));
    snapshot
}

fn reset_allocations() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocation_snapshot() -> AllocSnapshot {
    AllocSnapshot {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn parse_source(source: &str) -> source::SourceProgram {
    source::parse_ir(source, SourceLanguage::Hologram).unwrap()
}

struct SourceCase {
    name: &'static str,
    source: String,
    legacy: String,
}

impl SourceCase {
    fn linear_chain(nodes: usize) -> Self {
        Self {
            name: "native_linear512",
            source: native_linear_chain(nodes),
            legacy: legacy_linear_chain(nodes),
        }
    }

    fn large_constant(values: usize) -> Self {
        Self {
            name: "native_const4096",
            source: native_large_constant(values),
            legacy: legacy_large_constant(values),
        }
    }

    fn assert_archive_equivalent(&self) {
        let native = archive_from(&self.source);
        let legacy = archive_from(&self.legacy);
        assert_eq!(native, legacy);
    }
}

fn archive_from(source: &str) -> Vec<u8> {
    compile_from_source_language(
        source,
        SourceLanguage::Hologram,
        WittLevel::W32,
        BackendKind::Cpu,
    )
    .unwrap()
    .archive
}

fn native_linear_chain(nodes: usize) -> String {
    let mut source = String::from("input x: f32[1]\n");
    for index in 0..nodes {
        push_native_relu(&mut source, index);
    }
    source.push_str(&format!("output y{}\n", nodes - 1));
    source
}

fn push_native_relu(source: &mut String, index: usize) {
    let input = if index == 0 {
        String::from("x")
    } else {
        format!("y{}", index - 1)
    };
    source.push_str(&format!("let y{index}: f32[1] = relu({input})\n"));
}

fn legacy_linear_chain(nodes: usize) -> String {
    let mut source = String::from("input x :1\n");
    for index in 0..nodes {
        push_legacy_relu(&mut source, index);
    }
    source.push_str(&format!("output y{}\n", nodes - 1));
    source
}

fn push_legacy_relu(source: &mut String, index: usize) {
    let input = if index == 0 {
        String::from("x")
    } else {
        format!("y{}", index - 1)
    };
    source.push_str(&format!("op relu {input} :1 as=y{index}\n"));
}

fn native_large_constant(values: usize) -> String {
    format!("const w: f32[{values}] = [{}]\n", values_csv(values))
}

fn legacy_large_constant(values: usize) -> String {
    format!("const w :{values} = {}\n", values_csv(values))
}

fn values_csv(values: usize) -> String {
    (0..values)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

criterion_group!(benches, bench_source_lowering);
criterion_main!(benches);
