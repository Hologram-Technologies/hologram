//! End-to-end hologram pipeline demo.
//!
//! Walks the full current pipeline against the CPU backend:
//!
//! 1. Parse a line-oriented hologram source into a `Graph`.
//! 2. Compile the graph to a `.holo` archive (per-node Prism `CompileUnit`
//!    + completeness tower → certificate, lowered to `KernelCall`s).
//! 3. Inspect the archive's section table and decoded structure.
//! 4. Load the archive into an `InferenceSession` and execute it.
//! 5. Mint UOR-ADDR κ-labels for the model's parts and compose them into a
//!    single content address (decomposition → CS-G2 composition), verifying
//!    the replayable witness.
//!
//! Run:
//!
//! ```bash
//! cargo run -p hologram-cli --example pipeline
//! ```

use hologram_archive::address::{address_ring, compose_model};
use hologram_archive::{decoder, format::SectionKind, HoloLoader};
use hologram_backend::CpuBackend;
use hologram_compiler::{source, BackendKind, Compiler};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

/// A small activation chain: `y = sigmoid(gelu(relu(x)))`.
const SOURCE: &str = "\
# activation pipeline
input x
op relu x as=a
op gelu a as=b
op sigmoid b as=y
output y
";

fn main() {
    println!("=== hologram pipeline demo ===\n");

    // ── 1 + 2: parse → compile ────────────────────────────────────────
    let level = WittLevel::new(32);
    let graph = source::parse(SOURCE).expect("source parses");
    let compiled = Compiler::new(graph, BackendKind::Cpu, level)
        .compile()
        .expect("graph compiles");
    let archive = compiled.archive;
    let stats = compiled.stats;
    println!("compiled {} bytes", archive.len());
    println!(
        "  nodes={} levels={} validated={} cache_hits={} cache_misses={}",
        stats.total_nodes,
        stats.schedule_levels,
        stats.validated_units,
        stats.cache_hits,
        stats.cache_misses,
    );

    // ── 3: inspect the archive ────────────────────────────────────────
    let plan = HoloLoader::from_bytes(&archive)
        .expect("archive header verifies")
        .into_plan()
        .expect("archive plan decodes");
    println!("\narchive sections:");
    for s in plan.sections() {
        println!("  {:?} @ {} ({} bytes)", s.kind, s.offset, s.length);
    }
    if let Ok(calls_section) = plan.section(SectionKind::KernelCalls) {
        if let Ok(calls) = decoder::decode_calls(calls_section) {
            println!("kernel calls: {}", calls.len());
        }
    }

    // ── 4: load + execute ─────────────────────────────────────────────
    let backend = CpuBackend::<BufferArena>::new();
    let mut session = InferenceSession::load(&archive, backend).expect("session loads");
    let zeros = vec![0u8; 4096];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).expect("execution succeeds");
    println!(
        "\nexecution: {} input(s) → {} output(s)",
        inputs.len(),
        outputs.len()
    );
    for (i, o) in outputs.iter().enumerate() {
        println!("  output[{i}] = {} bytes", o.bytes.len());
    }

    // ── 5: UOR-ADDR content addressing + composition ──────────────────
    // Address two ring-element "parts" (Amendment-43 canonical bytes:
    // [witt_level] || le_coefficient) to κ-labels, then compose them into a
    // single model identity via the CS-G2 commutative product.
    let part_a = address_ring(&[1, 0x02, 0x01]).expect("ring element a addresses");
    let part_b = address_ring(&[2, 0x10, 0x20, 0x30]).expect("ring element b addresses");
    println!("\nUOR-ADDR κ-labels:");
    println!("  part a: {}", part_a.address.as_str());
    println!("  part b: {}", part_b.address.as_str());

    // Each κ-label carries a replayable TC-05 witness.
    assert_eq!(
        part_a.witness.verify().expect("witness a replays"),
        part_a.address
    );

    let model = compose_model(&[part_a.address, part_b.address]).expect("composes");
    println!("  model (a ∘ b): {}", model.as_str());
    // CS-G2 commutativity is structural: order does not matter.
    let model_rev = compose_model(&[part_b.address, part_a.address]).expect("composes");
    assert_eq!(model, model_rev);
    println!("  (order-independent: a ∘ b == b ∘ a ✓)");

    println!("\nOK");
}
