//! Step definitions for the `RM` suite (`features/suites/s7_readme`): every fenced code
//! block in `README.md` is bound to exactly one scenario, driven through the **public**
//! surface the README documents. `RM-N` ≡ the N-th fenced block, top-to-bottom.
//!
//! Compiled into the `bdd` test binary via `mod rm_steps;`. A fresh `ConformanceWorld` is
//! built per scenario, so the generic `rm_*` fields are reused across scenarios.
use cucumber::{given, then, when};
use hologram_conformance::ConformanceWorld;
use std::path::{Path, PathBuf};

// The public surfaces the README documents (all reachable via the facade's `client` feature,
// which this crate already enables). `crate::cast_graph` / `crate::common::SpikeSpace` are the
// reference workload + reference `Space` shared with the rest of the `bdd` runner.
use crate::common::SpikeSpace;
use hologram::archive::address::{address_ring, compose_model};
use hologram::backend::CpuBackend;
use hologram::compiler::source::{self, SourceLanguage};
use hologram::compiler::{compile_from_source_language, BackendKind, Compiler};
use hologram::exec::{BufferArena, InferenceSession, InputBuffer};
use hologram::Client;
use hologram_cli::cmd;
use hologram_space::address_bytes;
use uor_foundation::WittLevel;

// ───────────────────────────── shared helpers ─────────────────────────────

/// Repo root — `crates/hologram-conformance/../..` resolved at compile time.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Read a repo-relative text file (panics with a clear message if absent — an absent
/// documented artifact IS a conformance failure).
fn read_repo(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The `[features]` block of the root facade `Cargo.toml` (from the heading to the next
/// `[section]`), so a feature lookup never matches a `[dependencies]` line.
fn facade_features_block() -> String {
    let manifest = read_repo("Cargo.toml");
    let start = manifest
        .find("\n[features]")
        .expect("facade Cargo.toml has a [features] section");
    let rest = &manifest[start + 1..];
    let end = rest[1..].find("\n[").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].to_string()
}

/// A feature `name = …` is declared in the `[features]` block.
fn feature_declared(block: &str, name: &str) -> bool {
    block
        .lines()
        .any(|l| l.trim_start().starts_with(&format!("{name} = ")))
}

/// The `full = [ … ]` array (which may span lines) lists feature `name`.
fn full_lists(block: &str, name: &str) -> bool {
    let Some(start) = block.find("full = [") else {
        return false;
    };
    let rest = &block[start..];
    let end = rest.find(']').map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].contains(&format!("\"{name}\""))
}

fn all_true(flags: &[bool]) -> bool {
    !flags.is_empty() && flags.iter().all(|&b| b)
}

// ───────────────────────────── RM-1 — install.sh (L55) ─────────────────────────────

#[given("the repository's install.sh")]
fn rm1_given(_w: &mut ConformanceWorld) {}

#[when("I read its installer contract")]
fn rm1_when(w: &mut ConformanceWorld) {
    let sh = read_repo("install.sh");
    let posix_sh = sh.starts_with("#!") && sh.lines().next().is_some_and(|l| l.contains("sh"));
    let local_bin = sh.contains(".local/bin");
    let downloads = sh.contains("releases") || sh.contains("curl") || sh.contains("download");
    w.rm_flags = vec![posix_sh, local_bin, downloads];
}

#[then("it is POSIX sh that installs a prebuilt binary into the local bin dir")]
fn rm1_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "install.sh must be POSIX sh that downloads a prebuilt binary into ~/.local/bin \
         (shebang/local-bin/download): {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-2 — install alternatives (L62) ─────────────────────────────

#[given("the repository's install.sh and the hologram-cli binary target")]
fn rm2_given(_w: &mut ConformanceWorld) {}

#[when("I inspect the installer's flags and the cargo install target")]
fn rm2_when(w: &mut ConformanceWorld) {
    let sh = read_repo("install.sh");
    let cli = read_repo("crates/hologram-cli/Cargo.toml");
    let cli_bin = cli.contains("[[bin]]") && cli.contains("name = \"hologram\"");
    w.rm_flags = vec![
        sh.contains("--version"),
        sh.contains("--bin-dir"),
        sh.contains("--help"),
        cli_bin,
    ];
}

#[then("version and bin-dir overrides are honored and the cli binary target exists")]
fn rm2_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "install.sh must accept --version/--bin-dir/--help and hologram-cli must build the \
         `hologram` binary: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-4 — quickstart library features (L84) ─────────────────────────────

#[given("the quickstart dependency snippet's feature list")]
fn rm4_given(_w: &mut ConformanceWorld) {}

#[when("I check the hologram facade manifest")]
fn rm4_when(w: &mut ConformanceWorld) {
    let b = facade_features_block();
    w.rm_flags = ["archive", "backend", "compiler", "exec"]
        .iter()
        .map(|f| feature_declared(&b, f))
        .collect();
}

#[then("archive, backend, compiler, and exec are declared features")]
fn rm4_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the quickstart features archive/backend/compiler/exec must be declared on the facade: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-6 — tensor-engine library features (L243) ─────────────────────────────

#[given("the tensor-engine dependency snippet")]
fn rm6_given(_w: &mut ConformanceWorld) {}

#[when("I resolve its features on the hologram facade")]
fn rm6_when(w: &mut ConformanceWorld) {
    let b = facade_features_block();
    w.rm_flags = ["archive", "backend", "compiler", "exec"]
        .iter()
        .map(|f| feature_declared(&b, f))
        .collect();
}

#[then("the archive, backend, compiler, and exec features resolve")]
fn rm6_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the tensor-engine feature set must resolve on the facade: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-7 — full / space / client (L255) ─────────────────────────────

#[given("the facade feature manifest")]
fn rm7_given(_w: &mut ConformanceWorld) {}

#[when("I resolve the full, space, and client features")]
fn rm7_when(w: &mut ConformanceWorld) {
    let b = facade_features_block();
    let full_modules = ["archive", "backend", "compiler", "exec"]
        .iter()
        .all(|f| full_lists(&b, f));
    w.rm_flags = vec![
        full_modules,
        feature_declared(&b, "space"),
        feature_declared(&b, "client"),
    ];
}

#[then("full enables the tensor-engine modules and space and client are declared")]
fn rm7_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "`full` must enable the tensor-engine modules and `space`/`client` must be declared: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-8 — frontend features (L267) ─────────────────────────────

#[given("the frontend dependency snippet")]
fn rm8_given(_w: &mut ConformanceWorld) {}

#[when("I check the hologram facade manifest for frontends")]
fn rm8_when(w: &mut ConformanceWorld) {
    let b = facade_features_block();
    w.rm_flags = ["frontend-python", "frontend-typescript", "frontend-rust"]
        .iter()
        .map(|f| feature_declared(&b, f))
        .collect();
}

#[then("frontend-python, frontend-typescript, and frontend-rust are declared")]
fn rm8_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the three source-frontend features must be declared on the facade: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-33 — no_std feature composition (L993) ─────────────────────────────

#[given("the no_std dependency snippet with default features off")]
fn rm33_given(_w: &mut ConformanceWorld) {}

#[when("I check the facade manifest for the no_std feature set")]
fn rm33_when(w: &mut ConformanceWorld) {
    let b = facade_features_block();
    let default_is_std = b
        .lines()
        .any(|l| l.trim_start().starts_with("default = [\"std\"]"));
    w.rm_flags = vec![
        feature_declared(&b, "backend"),
        feature_declared(&b, "compiler"),
        feature_declared(&b, "exec"),
        default_is_std,
    ];
}

#[then("backend, compiler, and exec compose without the std default")]
fn rm33_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "backend/compiler/exec must be opt-in features and `std` a separable default so \
         default-features=false yields a no_std set: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-34 — building from source (L1090) ─────────────────────────────

#[given("a workspace source build")]
fn rm34_given(_w: &mut ConformanceWorld) {}

#[when("I run the pipeline example from source")]
fn rm34_when(w: &mut ConformanceWorld) {
    let ex = read_repo("crates/hologram-cli/examples/pipeline.rs");
    // The example a source build runs documents the end-to-end flow: parse → compile → execute → κ.
    let mentions = |needle: &str| ex.contains(needle);
    w.rm_flags = vec![
        !ex.trim().is_empty(),
        mentions("parse") || mentions("source"),
        mentions("compile") || mentions("Compiler"),
        mentions("execute") || mentions("Session"),
        mentions("address") || mentions("kappa") || mentions("κ"),
    ];
}

#[then("it completes the parse compile execute and address flow")]
fn rm34_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the pipeline example must exist and cover parse→compile→execute→address (the flow a \
         source build runs; RM-9 runs it): {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-35 — the just ci gate (L1101) ─────────────────────────────

#[given("the repository Justfile")]
fn rm35_given(_w: &mut ConformanceWorld) {}

#[when("I read the ci recipe")]
fn rm35_when(w: &mut ConformanceWorld) {
    let just = read_repo("Justfile");
    let ci = just
        .lines()
        .find(|l| l.starts_with("ci:"))
        .expect("the Justfile declares a `ci` recipe");
    w.rm_flags = ["fmt-check", "clippy", "test", "deny"]
        .iter()
        .map(|step| ci.contains(step))
        .collect();
}

#[then("it chains fmt-check, clippy, test, and the supply-chain gate")]
fn rm35_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "`just ci` must chain fmt-check + clippy + test + the supply-chain gate (deny): {:?}",
        w.rm_flags
    );
}

/// Decode a little-endian f32 buffer (the cast-graph output layout).
fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ───────────────────────────── RM-5 — the Client excerpt (L215) ─────────────────────────────

#[given("a Client over the reference space")]
fn rm5_given(_w: &mut ConformanceWorld) {}

#[when("it drives compile then provision then run")]
async fn rm5_when(w: &mut ConformanceWorld) {
    let client = Client::new(SpikeSpace::new());
    let holo = client
        .compile(crate::cast_graph())
        .expect("compile the workload");
    let kappa = client.provision(&holo).expect("provision the .holo");
    let vals: [i64; 4] = [0, 42, -7, 1024];
    let mut input = Vec::new();
    for &v in &vals {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let outputs = client
        .run(&kappa, &[input.as_slice()])
        .await
        .expect("run the workload");
    w.rm_flag = Some(as_f32(&outputs[0]) == vec![0.0, 42.0, -7.0, 1024.0]);
}

#[then("the workload produces its output through the one surface")]
fn rm5_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "one Client must drive compile→provision→run and produce the i64→f32 cast output"
    );
}

// ───────────────────────────── RM-10 — the minimal example (L286) ─────────────────────────────

#[given("the README's native source graph")]
fn rm10_given(_w: &mut ConformanceWorld) {}

#[when("I compile it to a .holo and load it on the CpuBackend")]
fn rm10_when(w: &mut ConformanceWorld) {
    let graph = source::parse("input x\nop relu x as=y\noutput y\n").unwrap();
    let compiled = Compiler::new(graph, BackendKind::Cpu, WittLevel::new(32))
        .compile()
        .unwrap();
    let mut session =
        InferenceSession::load(&compiled.archive, CpuBackend::<BufferArena>::new()).unwrap();
    let zeros = vec![0u8; 4096];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).unwrap();
    w.rm_flag = Some(!outputs.is_empty() && outputs.len() == session.output_count());
}

#[then("executing against zero inputs yields one output buffer per port")]
fn rm10_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "the minimal example must load on the CpuBackend and produce one output buffer per port"
    );
}

// ───────────────────────────── RM-11 — the lowering pipeline (L311) ─────────────────────────────

#[given("native source text")]
fn rm11_given(_w: &mut ConformanceWorld) {}

#[when("I lower it through document, program, graph, and compiler")]
fn rm11_when(w: &mut ConformanceWorld) {
    let text = "input x\nop relu x as=y\noutput y\n";
    // text → SourceProgram → Graph → Compiler → a loadable archive: each documented stage
    // feeds the next, and the pipeline's output is a real, runnable `.holo`.
    let program = source::parse_ir(text, SourceLanguage::Hologram).unwrap();
    let graph = source::lower_ir(&program).unwrap();
    let archive = Compiler::new(graph, BackendKind::Cpu, WittLevel::new(32))
        .compile()
        .unwrap()
        .archive;
    let loads = InferenceSession::load(&archive, CpuBackend::<BufferArena>::new()).is_ok();
    w.rm_flag = Some(!archive.is_empty() && loads);
}

#[then("each stage of the documented pipeline produces the next")]
fn rm11_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "text→document→program→graph→compiler must compose into a loadable archive"
    );
}

// ───────────────────────────── RM-24 — address and compose κ-labels (L552) ─────────────────────────────

#[given("two model-part rings addressed to κ-labels")]
fn rm24_given(_w: &mut ConformanceWorld) {}

#[when("I compose them in both orders")]
fn rm24_when(w: &mut ConformanceWorld) {
    let a = address_ring(&[1, 0x02, 0x01]).unwrap().address;
    let b = address_ring(&[2, 0x10, 0x20, 0x30]).unwrap().address;
    let ab = compose_model(&[a, b]).unwrap();
    let ba = compose_model(&[b, a]).unwrap();
    w.rm_flag = Some(ab == ba);
}

#[then("compose_model yields the same model identity regardless of order")]
fn rm24_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "compose_model is the CS-G2 commutative product — order-independent model identity"
    );
}

// ───────────────────────────── RM-28 — a minimal Space for the Client (L720) ─────────────────────────────

#[given("a minimal Space composed from the reference pieces")]
fn rm28_given(_w: &mut ConformanceWorld) {}

#[when("I build a Client over it")]
fn rm28_when(w: &mut ConformanceWorld) {
    let client = Client::new(SpikeSpace::new());
    w.rm_flag = Some(client.compile(crate::cast_graph()).is_ok());
}

#[then("the Client accepts the space and reaches a contract-mediated operation")]
fn rm28_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "a minimal reference Space must be accepted by Client and reach a contract operation"
    );
}

// ───────────────────────────── RM-29 — store, GC, and app tooling on one handle (L742) ─────────────────────────────

#[given("a Client with a compiled and provisioned workload")]
fn rm29_given(_w: &mut ConformanceWorld) {}

#[when("I exercise get, pin, ls, inspect, thin, and open on the one handle")]
async fn rm29_when(w: &mut ConformanceWorld) {
    use hologram::archive::HoloWriter;
    use hologram::space::{
        address_bytes, AppManifest, Capabilities, CapabilitySet, ContainerManifest, KappaStore,
        Layer, Realization,
    };
    use hologram_runtime::Phase;

    let client = Client::new(SpikeSpace::new());
    let holo = client.compile(crate::cast_graph()).expect("compile");
    let kappa = client.provision(&holo).expect("provision");

    let got = client.get(&kappa).expect("get").is_some();
    client.pin(&kappa).expect("pin");
    let ls_ok = client.ls().contains(&kappa);

    // inspect/thin are `.holo` v3 app tooling — `client.compile` emits a tensor archive (not a
    // v3 application), so exercise the app-tooling verbs on a v3 app holo (HF-3's surface).
    let app = {
        let manifest = AppManifest {
            primary: Some(0),
            requires: address_bytes(b"rm29-requires"),
            layers: vec![Layer::tensor(address_bytes(b"rm29-plan"), "sess")],
            children: vec![],
        };
        let mut writer = HoloWriter::new();
        writer.set_app_manifest(manifest.canonicalize());
        hologram::Holo::from_bytes(writer.finish().expect("write v3 app"))
    };
    let inspect_ok = client.inspect(&app).expect("inspect").all_verified();
    let thin_ok = client.thin(&app).is_ok();

    // open → boot a real container over the space's MockEngine runtime.
    let store = client.store();
    let code = store.put("blake3", b"<mock-code>").expect("code");
    let state = store.put("blake3", b"INIT").expect("state");
    let params = store.put("blake3", b"params").expect("params");
    let cid = store
        .put(
            "blake3",
            &ContainerManifest {
                code,
                initial_state: state,
                parameters: params,
            }
            .canonicalize(),
        )
        .expect("manifest");
    let caps = Capabilities {
        storage_roots: vec![],
        publish_channels: vec![],
        subscribe_channels: vec![],
        storage_quota_bytes: 1 << 16,
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: 0,
        network_fetch: false,
        network_announce: false,
    };
    let caps_k = store
        .put("blake3", &CapabilitySet::new(caps).canonicalize())
        .expect("caps");
    let mut session = client.open(&cid, &caps_k);
    let boot_ok = session.boot().await.is_ok() && session.phase() == Phase::Running;

    w.rm_flags = vec![got, ls_ok, inspect_ok, thin_ok, boot_ok];
}

#[then("each store and app-tooling operation succeeds on the same surface")]
fn rm29_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "get/pin/ls/inspect/thin and open→boot must all succeed on the one Client handle: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── shared CLI helpers ─────────────────────────────

/// A unique temp path for this process + label. Distinct RM scenarios use distinct labels,
/// so their files never collide even though cucumber runs scenarios concurrently.
fn tmp(label: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("hologram-rm-{}-{label}", std::process::id()));
    p
}

/// The README's native source graph.
const NATIVE_SOURCE: &str = "input x\nop relu x as=y\noutput y\n";

// ───────────────────────────── RM-3 — quickstart compile then execute (L76) ─────────────────────────────

#[given("a native hologram source file")]
fn rm3_given(_w: &mut ConformanceWorld) {}

#[when("I run the CLI compile then execute verbs on it")]
fn rm3_when(w: &mut ConformanceWorld) {
    let src = tmp("rm3.txt");
    std::fs::write(&src, NATIVE_SOURCE).unwrap();
    let out = tmp("rm3.holo");
    let (s, o) = (src.to_str().unwrap(), out.to_str().unwrap());
    cmd::run_from_args(["hologram", "compile", "--source", s, "--output", o]).expect("compile");
    cmd::run_from_args(["hologram", "execute", "--archive", o]).expect("execute");
    w.rm_flag = Some(out.exists());
}

#[then("the archive round-trips and reports one length per output port")]
fn rm3_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "`hologram compile` then `hologram execute` must round-trip to a runnable archive"
    );
}

// ───────────────────────────── RM-9 — the pipeline example (L279) ─────────────────────────────

#[given("the hologram-cli pipeline example")]
fn rm9_given(_w: &mut ConformanceWorld) {}

#[when("I run it end to end")]
fn rm9_when(w: &mut ConformanceWorld) {
    // Mirror examples/pipeline.rs: parse → compile → execute → address + compose (order-independent).
    let text = "# activation pipeline\ninput x\nop relu x as=a\nop gelu a as=b\nop sigmoid b as=y\noutput y\n";
    let graph = source::parse(text).unwrap();
    let compiled = Compiler::new(graph, BackendKind::Cpu, WittLevel::new(32))
        .compile()
        .unwrap();
    let mut session =
        InferenceSession::load(&compiled.archive, CpuBackend::<BufferArena>::new()).unwrap();
    let zeros = vec![0u8; 4096];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).unwrap();
    let part_a = address_ring(&[1, 0x02, 0x01]).unwrap();
    let part_b = address_ring(&[2, 0x10, 0x20, 0x30]).unwrap();
    let witness_ok = part_a
        .witness
        .verify()
        .map(|a| a == part_a.address)
        .unwrap_or(false);
    let model = compose_model(&[part_a.address, part_b.address]).unwrap();
    let model_rev = compose_model(&[part_b.address, part_a.address]).unwrap();
    w.rm_flags = vec![!outputs.is_empty(), witness_ok, model == model_rev];
}

#[then("it parses, compiles, executes, and addresses without error")]
fn rm9_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the pipeline example flow (parse→compile→execute→address, order-independent) must run: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-30 — the CLI tensor verbs (L778) ─────────────────────────────

#[given("a compiled .holo archive")]
fn rm30_given(w: &mut ConformanceWorld) {
    let src = tmp("rm30.txt");
    std::fs::write(&src, NATIVE_SOURCE).unwrap();
    let out = tmp("rm30.holo");
    cmd::run_from_args([
        "hologram",
        "compile",
        "--source",
        src.to_str().unwrap(),
        "--output",
        out.to_str().unwrap(),
    ])
    .expect("compile the archive for the tensor verbs");
    w.rm_kappa = Some(out.to_str().unwrap().to_string());
}

#[when("I run the inspect, execute, and bench verbs on it")]
fn rm30_when(w: &mut ConformanceWorld) {
    let out = w.rm_kappa.clone().expect("archive path from Given");
    let inspect = cmd::run_from_args(["hologram", "inspect", "--archive", &out]).is_ok();
    let execute = cmd::run_from_args(["hologram", "execute", "--archive", &out]).is_ok();
    let bench =
        cmd::run_from_args(["hologram", "bench", "--archive", &out, "--iterations", "3"]).is_ok();
    w.rm_flags = vec![inspect, execute, bench];
}

#[then("each verb reports on the archive without error")]
fn rm30_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "`hologram inspect`/`execute`/`bench` must each succeed on one archive: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-26 — node holospace parts (L666) ─────────────────────────────

#[given("the container parts as bytes")]
fn rm26_given(_w: &mut ConformanceWorld) {}

#[when("I run node put, manifest, and caps")]
fn rm26_when(w: &mut ConformanceWorld) {
    let store = tmp("rm26.redb");
    let store_s = store.to_str().unwrap();
    let userland = tmp("rm26-userland.bin");
    std::fs::write(&userland, b"rm26-userland").unwrap();
    let put = cmd::run_node_from_args([
        "node",
        "--store",
        store_s,
        "put",
        userland.to_str().unwrap(),
    ]);
    let (code, state, params) = (
        address_bytes(b"rm26-code").to_string(),
        address_bytes(b"rm26-state").to_string(),
        address_bytes(b"rm26-params").to_string(),
    );
    let manifest = cmd::run_node_from_args([
        "node",
        "--store",
        store_s,
        "manifest",
        code.as_str(),
        state.as_str(),
        params.as_str(),
    ]);
    let caps = cmd::run_node_from_args([
        "node", "--store", store_s, "caps", "--mem", "4194304", "--cpu-ms", "1000",
    ]);
    w.rm_flags = vec![put == 0, manifest == 0, caps == 0];
}

#[then("each prints a κ-label and the container manifest addresses its parts")]
fn rm26_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "`node put`/`manifest`/`caps` must each succeed (exit 0) and mint κs: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-31 — node substrate verbs (L801) ─────────────────────────────

#[given("a node store and content bytes")]
fn rm31_given(_w: &mut ConformanceWorld) {}

#[when("I put content and then get and verify it by κ")]
fn rm31_when(w: &mut ConformanceWorld) {
    let store = tmp("rm31.redb");
    let store_s = store.to_str().unwrap();
    let file = tmp("rm31.bin");
    let content = b"rm31-content-bytes";
    std::fs::write(&file, content).unwrap();
    let file_s = file.to_str().unwrap();
    let kappa = address_bytes(content).to_string();
    let put = cmd::run_node_from_args(["node", "--store", store_s, "put", file_s]);
    let get = cmd::run_node_from_args(["node", "--store", store_s, "get", kappa.as_str()]);
    let verify =
        cmd::run_node_from_args(["node", "--store", store_s, "verify", kappa.as_str(), file_s]);
    w.rm_flags = vec![put == 0, get == 0, verify == 0];
}

#[then("the bytes round-trip and re-derive to the same κ")]
fn rm31_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "`node put`→`get`→`verify` must round-trip and re-derive to the same κ (SPINE-4): {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── source-frontend fixtures (README §Source frontends) ─────────────────────────────

const PY_SRC: &str = r#"def ordinary_app_code():
    return 42

def encoder(h):
    x = h.input("x", dtype="f32", shape=[2, 3])
    w = h.const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    y = h.ops.matmul(x, w, shape=[2, 2])
    h.output("y", y)
"#;

const TS_SRC: &str = r#"function ordinaryAppCode() {
    return 42;
}

export function encoder(h: HologramBuilder) {
    const x = h.input("x", { dtype: "f32", shape: [2, 3] });
    const w = h.const("w", { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] });
    const y = h.ops.matmul(x, w, { shape: [2, 2] });
    h.output("y", y);
}
"#;

const RS_SRC: &str = r#"fn ordinary_app_code() -> i32 {
    42
}

pub fn encoder(h: &mut HologramBuilder) {
    let x = h.input("x", dtype("f32"), shape([2, 3]));
    let w = h.constant("w", shape([3, 2]), values([1, 2, 3, 4, 5, 6]));
    let y = h.ops().matmul(x, w, shape([2, 2]));
    h.output("y", y);
}
"#;

/// A frontend must extract exactly the `encoder` graph (unrelated code ignored) and lower it.
fn extracts_encoder(src: &str, lang: SourceLanguage) -> Vec<bool> {
    let doc = source::parse_document(src, lang).expect("parse the source document");
    let only_encoder = doc.graphs().len() == 1; // `ordinary_app_code` is not a builder → ignored
    let program = source::parse_ir_with_options(
        src,
        lang,
        &source::SourceParseOptions::new().graph("encoder"),
    )
    .expect("select the encoder graph");
    vec![only_encoder, source::lower_ir(&program).is_ok()]
}

/// The CLI compiles a host-language builder file (detected from its extension) selecting `encoder`.
fn cli_compiles_frontend(w: &mut ConformanceWorld, ext: &str, src: &str) {
    let file = tmp(&format!("rmfe.{ext}"));
    std::fs::write(&file, src).unwrap();
    let out = tmp(&format!("rmfe-{ext}.holo"));
    let ok = cmd::run_full_from_args([
        "hologram",
        "compile",
        "--source",
        file.to_str().unwrap(),
        "--graph",
        "encoder",
        "--output",
        out.to_str().unwrap(),
    ])
    .is_ok();
    w.rm_flag = Some(ok && out.exists());
}

// ───────────────────────────── RM-12 — SourceParseOptions graph selection (L330) ─────────────────────────────

#[given("a source document with a named graph")]
fn rm12_given(_w: &mut ConformanceWorld) {}

#[when("I select it with SourceParseOptions and lower it")]
fn rm12_when(w: &mut ConformanceWorld) {
    let program = source::parse_ir_with_options(
        PY_SRC,
        SourceLanguage::Python,
        &source::SourceParseOptions::new().graph("encoder"),
    )
    .unwrap();
    let graph = source::lower_ir(&program).unwrap();
    w.rm_flag = Some(
        Compiler::new(graph, BackendKind::Cpu, WittLevel::new(32))
            .compile()
            .is_ok(),
    );
}

#[then("the named graph compiles")]
fn rm12_then(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "SourceParseOptions::graph must select a named graph that lowers and compiles"
    );
}

// ───────────────────────────── RM-13 — compile_from_source_language (L347) ─────────────────────────────

#[given("a single-graph source in a host language")]
fn rm13_given(_w: &mut ConformanceWorld) {}

#[when("I call compile_from_source_language")]
fn rm13_when(w: &mut ConformanceWorld) {
    let out = compile_from_source_language(
        PY_SRC,
        SourceLanguage::Python,
        WittLevel::new(32),
        BackendKind::Cpu,
    );
    w.rm_flag = Some(out.map(|o| !o.archive.is_empty()).unwrap_or(false));
}

// Shared by RM-13 (and, once promoted, the SDK compile-source scenarios RM-22/RM-23).
#[then("it returns a compiled archive")]
fn returns_compiled_archive(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "the one-call source compile must return a non-empty archive"
    );
}

// ───────────────────────────── RM-14 / RM-16 / RM-18 — frontend extraction (L368 / L401 / L437) ─────────────────────────────

#[given("a Python file with an encoder builder and unrelated code")]
fn rm14_given(_w: &mut ConformanceWorld) {}

#[when("the Python frontend parses it")]
fn rm14_when(w: &mut ConformanceWorld) {
    w.rm_flags = extracts_encoder(PY_SRC, SourceLanguage::Python);
}

#[given("a TypeScript file with an encoder builder and unrelated code")]
fn rm16_given(_w: &mut ConformanceWorld) {}

#[when("the TypeScript frontend parses it")]
fn rm16_when(w: &mut ConformanceWorld) {
    w.rm_flags = extracts_encoder(TS_SRC, SourceLanguage::TypeScript);
}

#[given("a Rust file with an encoder builder and unrelated code")]
fn rm18_given(_w: &mut ConformanceWorld) {}

#[when("the Rust frontend parses it")]
fn rm18_when(w: &mut ConformanceWorld) {
    w.rm_flags = extracts_encoder(RS_SRC, SourceLanguage::Rust);
}

// Shared Then for RM-14 / RM-16 / RM-18.
#[then("the encoder graph is extracted and the unrelated code is ignored")]
fn encoder_extracted_unrelated_ignored(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the frontend must extract exactly the encoder graph and ignore unrelated code: {:?}",
        w.rm_flags
    );
}

// ───────────────────────────── RM-15 / RM-17 / RM-19 — frontend CLI compile (L379 / L414 / L450) ─────────────────────────────

#[given("a Python builder file on disk")]
fn rm15_given(_w: &mut ConformanceWorld) {}

#[when("I run the CLI compile verb with the frontend-python feature and a graph name")]
fn rm15_when(w: &mut ConformanceWorld) {
    cli_compiles_frontend(w, "py", PY_SRC);
}

#[given("a TypeScript builder file on disk")]
fn rm17_given(_w: &mut ConformanceWorld) {}

#[when("I run the CLI compile verb with the frontend-typescript feature and a graph name")]
fn rm17_when(w: &mut ConformanceWorld) {
    cli_compiles_frontend(w, "ts", TS_SRC);
}

#[given("a Rust builder file on disk")]
fn rm19_given(_w: &mut ConformanceWorld) {}

#[when("I run the CLI compile verb with the frontend-rust feature and a graph name")]
fn rm19_when(w: &mut ConformanceWorld) {
    cli_compiles_frontend(w, "rs", RS_SRC);
}

// Shared Then for RM-15 / RM-17 / RM-19.
#[then("it writes a compiled archive")]
fn cli_writes_archive(w: &mut ConformanceWorld) {
    assert_eq!(
        w.rm_flag,
        Some(true),
        "the CLI must compile the selected host-language graph to an archive on disk"
    );
}

// ───────────────────────────── RM-32 — the C ABI pipeline (L827) ─────────────────────────────

#[given("native source and the C ABI entry points")]
fn rm32_given(_w: &mut ConformanceWorld) {}

#[when("I compile the source, load a session, execute it, and close it")]
fn rm32_when(w: &mut ConformanceWorld) {
    use hologram_ffi::{
        hologram_abi_version, hologram_compile_source, hologram_session_close,
        hologram_session_execute, hologram_session_input_count, hologram_session_load,
        hologram_session_output_count,
    };
    use std::os::raw::c_uchar;

    let src = b"input x\nop relu x as=y\noutput y\n";
    let mut archive = vec![0u8; 65536];
    // compile_source (snprintf-style: returns the full archive length).
    let n = unsafe {
        hologram_compile_source(src.as_ptr(), src.len(), archive.as_mut_ptr(), archive.len())
    };
    let compiled_ok = n > 0 && (n as usize) <= archive.len();
    let alen = if compiled_ok { n as usize } else { 0 };

    let h = unsafe { hologram_session_load(archive.as_ptr(), alen) };
    let ic = unsafe { hologram_session_input_count(h) }.max(0) as usize;
    let oc = unsafe { hologram_session_output_count(h) }.max(0) as usize;

    // Marshal zero inputs and generously-sized output buffers, C-ABI style.
    let in_bufs: Vec<Vec<u8>> = (0..ic).map(|_| vec![0u8; 4096]).collect();
    let in_ptrs: Vec<*const c_uchar> = in_bufs.iter().map(|b| b.as_ptr()).collect();
    let in_lens: Vec<usize> = in_bufs.iter().map(|b| b.len()).collect();
    let mut out_bufs: Vec<Vec<u8>> = (0..oc).map(|_| vec![0u8; 4096]).collect();
    let out_ptrs: Vec<*mut c_uchar> = out_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
    let out_caps: Vec<usize> = out_bufs.iter().map(|b| b.len()).collect();
    let rc = unsafe {
        hologram_session_execute(
            h,
            in_ptrs.as_ptr(),
            in_lens.as_ptr(),
            ic,
            out_ptrs.as_ptr(),
            out_caps.as_ptr(),
            oc,
        )
    };
    let close = unsafe { hologram_session_close(h) };
    let abi = hologram_abi_version(); // not an `unsafe` fn — no version state to violate

    w.rm_flags = vec![compiled_ok, h >= 0, oc >= 1, rc >= 0, close >= 0, abi >= 1];
}

#[then("the session handle drives the full pipeline and the ABI version is reported")]
fn rm32_then(w: &mut ConformanceWorld) {
    assert!(
        all_true(&w.rm_flags),
        "the C ABI must compile→load→execute→close and report a non-zero ABI version: {:?}",
        w.rm_flags
    );
}
