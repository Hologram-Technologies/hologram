//! CLI subcommands.

use clap::{Parser, Subcommand};
use hologram_compiler::error::CompileError;
use hologram_compiler::source::{self, SourceLanguage};
use hologram_compiler::{BackendKind, Compiler};
use hologram_graph::Graph;
use prism::vocabulary::WittLevel;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "hologram", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Parser, Debug)]
#[command(name = "hologram", version)]
struct CliArgs {
    #[command(subcommand)]
    command: CommandArgs,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Compile a hologram source file (or an empty graph if no source) to a `.holo` archive.
    Compile {
        #[arg(long, default_value = "cpu")]
        backend: String,
        #[arg(long, default_value_t = 32)]
        witt_level: u32,
        /// Input source file (line-oriented hologram-source). Optional — when omitted,
        /// the CLI compiles an empty graph.
        #[arg(long)]
        source: Option<std::path::PathBuf>,
        /// Output archive path.
        #[arg(long)]
        output: std::path::PathBuf,
        /// Skip the warm-start fold (WS-2). By default the CLI materializes
        /// the constant-only cone into the archive so the runtime cache is
        /// never cold; pass `--no-warm` to emit the labels-only lattice.
        #[arg(long)]
        no_warm: bool,
    },
    /// Execute a `.holo` archive against the CPU backend with zero-byte inputs.
    /// Returns the byte length of each declared output port.
    Execute {
        #[arg(long)]
        archive: std::path::PathBuf,
    },
    /// Inspect a `.holo` archive's section table.
    Inspect {
        #[arg(long)]
        archive: std::path::PathBuf,
    },
    /// Micro-bench: run an archive `iterations` times against zero inputs and
    /// report wall-clock per iteration. Useful for quick A/B comparisons.
    Bench {
        #[arg(long)]
        archive: std::path::PathBuf,
        #[arg(long, default_value_t = 100)]
        iterations: u32,
    },
}

#[derive(Subcommand, Debug)]
enum CommandArgs {
    /// Compile a hologram source file (or an empty graph if no source) to a `.holo` archive.
    Compile {
        #[arg(long, default_value = "cpu")]
        backend: String,
        #[arg(long, default_value_t = 32)]
        witt_level: u32,
        #[arg(long)]
        source: Option<std::path::PathBuf>,
        /// Source language (`hologram`, `python`, `typescript`, `rust`, or `auto`).
        #[arg(long, value_name = "LANG")]
        source_language: Option<String>,
        /// Graph name to compile when a source file contains multiple graph regions.
        #[arg(long, value_name = "NAME")]
        graph: Option<String>,
        #[arg(long)]
        output: std::path::PathBuf,
        #[arg(long)]
        no_warm: bool,
    },
    /// Execute a `.holo` archive against the CPU backend with zero-byte inputs.
    Execute {
        #[arg(long)]
        archive: std::path::PathBuf,
    },
    /// Inspect a `.holo` archive's section table.
    Inspect {
        #[arg(long)]
        archive: std::path::PathBuf,
    },
    /// Micro-bench: run an archive `iterations` times against zero inputs.
    Bench {
        #[arg(long)]
        archive: std::path::PathBuf,
        #[arg(long, default_value_t = 100)]
        iterations: u32,
    },
    /// Run a deployment-substrate node command (κ-label store / route + serve) — the node
    /// CLI unified into the one `hologram` binary (D13).
    Node(crate::node::NodeCli),
    /// `.holo` v3 application tooling (spec `refactor/03`): inspect an app's layers + certificates,
    /// or convert between fat and thin packaging without changing the app κ.
    App(AppCli),
    /// Network tooling (spec `refactor/04`): create a Network realization (the VPC analogue) or show
    /// one. Membership/policy/key are κ-addressed — a member/policy/key is content, named by its κ.
    Network(NetworkCli),
}

/// `hologram network <subcommand>` — network (VPC-analogue) tooling.
#[derive(clap::Args, Debug)]
pub struct NetworkCli {
    #[command(subcommand)]
    command: NetworkCommand,
}

#[derive(Subcommand, Debug)]
enum NetworkCommand {
    /// Create a Network realization: its membership is the κ of each `--member` content file, its
    /// policy the κ of the `--policy` file, and its tier `--tier`. `--key` (the κ of a symmetric-key
    /// file) is required for `private` and rejected otherwise. Writes the realization to `--output`.
    Create {
        /// Founding-member content file (e.g. an operator's attestation key); repeatable — its κ is
        /// the member. At least one.
        #[arg(long = "member", required = true)]
        members: Vec<std::path::PathBuf>,
        /// Policy CapabilitySet content file — its κ is the network policy.
        #[arg(long)]
        policy: std::path::PathBuf,
        /// Tier: `public`, `restricted`, or `private`.
        #[arg(long, default_value = "restricted")]
        tier: String,
        /// Symmetric-key content file (Private tier only) — its κ is the network key.
        #[arg(long)]
        key: Option<std::path::PathBuf>,
        /// Output path for the Network realization's canonical bytes.
        #[arg(long)]
        output: std::path::PathBuf,
    },
    /// Show a Network realization: its κ, tier, membership κs, policy κ, and key binding.
    Show {
        #[arg(long)]
        network: std::path::PathBuf,
    },
    /// Delegate membership: mint a `Delegation` realization granting an attenuated `--child`
    /// CapabilitySet from a `--parent` CapabilitySet. **Refuses amplification** (child ⊄ parent) —
    /// attenuation only (law 5). Both inputs are CapabilitySet realization files.
    Delegate {
        /// Parent CapabilitySet file (the delegating member's authority).
        #[arg(long)]
        parent: std::path::PathBuf,
        /// Child CapabilitySet file (the new member's attenuated authority).
        #[arg(long)]
        child: std::path::PathBuf,
        /// Output path for the Delegation realization.
        #[arg(long)]
        output: std::path::PathBuf,
    },
}

/// `hologram app <subcommand>` — `.holo` v3 application tooling.
#[derive(clap::Args, Debug)]
pub struct AppCli {
    #[command(subcommand)]
    command: AppCommand,
}

#[derive(Subcommand, Debug)]
enum AppCommand {
    /// Inspect a `.holo` v3 application: its identity κ, primary layer, per-layer descriptors, and
    /// children. Never strips certificates (spec 03 §per-layer certificates).
    Inspect {
        #[arg(long)]
        archive: std::path::PathBuf,
    },
    /// Convert a `.holo` to a **thin** archive: manifest + certificates only, dropping embedded
    /// content — layers resolve through the store/sync at load. The app κ is **unchanged**.
    Thin {
        /// Input `.holo`.
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output thin `.holo`.
        #[arg(long)]
        output: std::path::PathBuf,
    },
    /// Convert a `.holo` to a **fat** (self-contained) archive: embed every layer/closure κ
    /// resolvable from the `--store` as a content blob. The app κ is **unchanged**; κs absent from
    /// the store stay unresolved (re-run once synced).
    Fat {
        /// Input `.holo`.
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output fat `.holo`.
        #[arg(long)]
        output: std::path::PathBuf,
        /// A `NativeKappaStore` (redb file) to resolve layer content from.
        #[arg(long)]
        store: std::path::PathBuf,
    },
}

/// Parse command-line arguments from the process environment and run the CLI.
pub fn run_from_env() -> Result<(), CompileError> {
    run_args(CliArgs::parse())
}

/// Run the tensor verbs (`compile` / `execute` / `inspect` / `bench`) from an explicit
/// argument vector (`argv[0]` = program name), returning `Result` instead of touching the
/// process environment — the in-process entry point conformance tests drive.
pub fn run_from_args<I, T>(args: I) -> Result<(), CompileError>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::try_parse_from(args).map_err(|_| CompileError::SourceParse("cli args"))?;
    run(cli)
}

/// Run the FULL CLI from an explicit argument vector — including `compile`'s `--graph` /
/// `--source-language` options that the tensor-only [`run_from_args`] omits. Returns `Result`
/// for `compile` / `execute` / `inspect` / `bench` / `app` / `network`; the `node` arm still
/// exits the process, so drive node with [`run_node_from_args`].
pub fn run_full_from_args<I, T>(args: I) -> Result<(), CompileError>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_args(CliArgs::try_parse_from(args).map_err(|_| CompileError::SourceParse("cli args"))?)
}

/// Run the `node` verb group from an explicit argument vector, returning its exit code
/// (`0` = success) instead of exiting the process — the in-process entry conformance tests
/// drive (the unified CLI's node arm exits with this same code).
pub fn run_node_from_args<I, T>(args: I) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    #[derive(Parser)]
    struct NodeWrap {
        #[command(flatten)]
        cli: crate::node::NodeCli,
    }
    match NodeWrap::try_parse_from(args) {
        Ok(w) => crate::node::run(w.cli),
        Err(_) => 2,
    }
}

pub fn run(cli: Cli) -> Result<(), CompileError> {
    match cli.command {
        Command::Compile {
            backend,
            witt_level,
            source,
            output,
            no_warm,
        } => compile_command(CompileArgs {
            backend,
            witt_level,
            source,
            source_language: None,
            graph_name: None,
            output,
            no_warm,
        }),
        Command::Execute { archive } => {
            let bytes =
                std::fs::read(&archive).map_err(|_| CompileError::SourceParse("read archive"))?;
            let backend: hologram_compute::CpuBackend<hologram_exec::BufferArena> =
                hologram_compute::CpuBackend::new();
            let mut session = hologram_exec::InferenceSession::load(&bytes, backend)
                .map_err(|_| CompileError::SourceParse("load archive"))?;
            let zeros = zero_inputs_for(&session);
            let inputs: Vec<hologram_exec::InputBuffer> = zeros
                .iter()
                .map(|b| hologram_exec::InputBuffer { bytes: b })
                .collect();
            let outputs = session
                .execute(&inputs)
                .map_err(|_| CompileError::SourceParse("execute"))?;
            for (i, o) in outputs.iter().enumerate() {
                println!("output[{i}] = {} bytes", o.bytes.len());
            }
            Ok(())
        }
        Command::Inspect { archive } => {
            let bytes =
                std::fs::read(&archive).map_err(|_| CompileError::SourceParse("read archive"))?;
            let plan = hologram_archive::HoloLoader::from_bytes(&bytes)
                .map_err(CompileError::Archive)?
                .into_plan()
                .map_err(CompileError::Archive)?;
            println!("archive: {} bytes", bytes.len());
            for s in plan.sections() {
                println!("  section {:?} @ {} ({} bytes)", s.kind, s.offset, s.length);
            }
            // Decode + show kernel-call and exec-plan structure.
            if let Ok(calls_section) =
                plan.section(hologram_archive::format::SectionKind::KernelCalls)
            {
                if let Ok(calls) = hologram_archive::decoder::decode_calls(calls_section) {
                    println!("kernel_calls: {}", calls.len());
                }
            }
            if let Ok(exec_section) = plan.section(hologram_archive::format::SectionKind::ExecPlan)
            {
                if let Ok(plan) = hologram_archive::decode_exec_plan(exec_section) {
                    println!(
                        "exec_plan: {} levels, max_width={}",
                        plan.len(),
                        plan.iter().map(|l| l.len()).max().unwrap_or(0)
                    );
                }
            }
            Ok(())
        }
        Command::Bench {
            archive,
            iterations,
        } => {
            let bytes =
                std::fs::read(&archive).map_err(|_| CompileError::SourceParse("read archive"))?;
            let backend: hologram_compute::CpuBackend<hologram_exec::BufferArena> =
                hologram_compute::CpuBackend::new();
            let mut session = hologram_exec::InferenceSession::load(&bytes, backend)
                .map_err(|_| CompileError::SourceParse("load archive"))?;
            let zeros = zero_inputs_for(&session);
            let inputs: Vec<hologram_exec::InputBuffer> = zeros
                .iter()
                .map(|b| hologram_exec::InputBuffer { bytes: b })
                .collect();
            // Warmup.
            let _ = session.execute(&inputs);
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                session
                    .execute(&inputs)
                    .map_err(|_| CompileError::SourceParse("execute"))?;
            }
            let elapsed = start.elapsed();
            let per = elapsed / iterations.max(1);
            println!(
                "bench: {} iterations in {:?} ({:?}/iter, {} kernel calls, {} schedule levels)",
                iterations,
                elapsed,
                per,
                session.kernel_count(),
                session.schedule_levels(),
            );
            Ok(())
        }
    }
}

fn run_args(cli: CliArgs) -> Result<(), CompileError> {
    match cli.command {
        CommandArgs::Compile {
            backend,
            witt_level,
            source,
            source_language,
            graph: graph_name,
            output,
            no_warm,
        } => compile_command(CompileArgs {
            backend,
            witt_level,
            source,
            source_language,
            graph_name,
            output,
            no_warm,
        }),
        CommandArgs::Execute { archive } => run(Cli {
            command: Command::Execute { archive },
        }),
        CommandArgs::Inspect { archive } => run(Cli {
            command: Command::Inspect { archive },
        }),
        CommandArgs::Bench {
            archive,
            iterations,
        } => run(Cli {
            command: Command::Bench {
                archive,
                iterations,
            },
        }),
        // The node subcommand is self-contained (its own store/engine/transport + error
        // handling); it prints and returns a process exit code, so we exit directly.
        CommandArgs::Node(node_cli) => std::process::exit(i32::from(crate::node::run(node_cli))),
        CommandArgs::App(app_cli) => run_app(app_cli),
        CommandArgs::Network(net_cli) => run_network(net_cli),
    }
}

fn run_network(net_cli: NetworkCli) -> Result<(), CompileError> {
    match net_cli.command {
        NetworkCommand::Create {
            members,
            policy,
            tier,
            key,
            output,
        } => net_create(&members, &policy, &tier, key.as_deref(), &output),
        NetworkCommand::Show { network } => net_show(&network),
        NetworkCommand::Delegate {
            parent,
            child,
            output,
        } => net_delegate(&parent, &child, &output),
    }
}

fn net_delegate(parent: &Path, child: &Path, output: &Path) -> Result<(), CompileError> {
    use hologram_space::{address_bytes, CapabilitySet, Delegation, Realization};
    let parent_bytes =
        std::fs::read(parent).map_err(|_| CompileError::SourceParse("read parent capset"))?;
    let child_bytes =
        std::fs::read(child).map_err(|_| CompileError::SourceParse("read child capset"))?;
    let parent_caps = CapabilitySet::to_capabilities(&parent_bytes)
        .map_err(|_| CompileError::SourceParse("parent is not a CapabilitySet realization"))?;
    let child_caps = CapabilitySet::to_capabilities(&child_bytes)
        .map_err(|_| CompileError::SourceParse("child is not a CapabilitySet realization"))?;
    // Attenuation only (law 5): the child's authority must be a subset of the parent's.
    if !parent_caps.admits(&child_caps) {
        return Err(CompileError::SourceParse(
            "delegation would amplify authority (child is not a subset of parent) — refused",
        ));
    }
    let delegation = Delegation {
        parent_caps: address_bytes(&parent_bytes),
        child_caps: address_bytes(&child_bytes),
    };
    std::fs::write(output, delegation.canonicalize())
        .map_err(|_| CompileError::SourceParse("write delegation"))?;
    println!(
        "delegated (attenuated) parent κ={} → child κ={} at {}",
        String::from_utf8_lossy(delegation.parent_caps.as_array()),
        String::from_utf8_lossy(delegation.child_caps.as_array()),
        output.display()
    );
    Ok(())
}

fn parse_tier(s: &str) -> Result<hologram_space::NetworkTier, CompileError> {
    use hologram_space::NetworkTier;
    match s {
        "public" => Ok(NetworkTier::Public),
        "restricted" => Ok(NetworkTier::Restricted),
        "private" => Ok(NetworkTier::Private),
        _ => Err(CompileError::SourceParse(
            "unknown tier (expected: public, restricted, private)",
        )),
    }
}

/// The κ of a content file — a member / policy / key is content, named by its κ (SPINE-1).
fn kappa_of_file(path: &Path) -> Result<hologram_space::KappaLabel71, CompileError> {
    let bytes = std::fs::read(path).map_err(|_| CompileError::SourceParse("read content file"))?;
    Ok(hologram_space::address_bytes(&bytes))
}

fn net_create(
    members: &[std::path::PathBuf],
    policy: &Path,
    tier: &str,
    key: Option<&Path>,
    output: &Path,
) -> Result<(), CompileError> {
    use hologram_space::{NetworkTier, Realization};
    let tier = parse_tier(tier)?;
    let membership = members
        .iter()
        .map(|p| kappa_of_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    let policy_kappa = kappa_of_file(policy)?;
    let key_ref = match key {
        Some(p) => Some(kappa_of_file(p)?),
        None => None,
    };
    let network = hologram_space::Network {
        membership,
        policy: policy_kappa,
        parent: None,
        tier,
        key_ref,
    };
    // Private ⟺ a bound key — reject a key on an unencrypted tier (false confidentiality) or a
    // Private network with no key.
    if !network.key_binding_ok() {
        return Err(CompileError::SourceParse(match tier {
            NetworkTier::Private => "private tier requires --key",
            _ => "--key is only valid for the private tier",
        }));
    }
    let bytes = network.canonicalize();
    std::fs::write(output, &bytes).map_err(|_| CompileError::SourceParse("write network"))?;
    println!(
        "created {tier:?} network κ={} ({} member(s)) → {}",
        String::from_utf8_lossy(network.kappa().as_array()),
        network.membership.len(),
        output.display()
    );
    Ok(())
}

fn net_show(network: &Path) -> Result<(), CompileError> {
    use hologram_space::Network;
    let bytes = std::fs::read(network).map_err(|_| CompileError::SourceParse("read network"))?;
    let net = Network::decode(&bytes)
        .map_err(|_| CompileError::SourceParse("malformed network realization"))?;
    println!(
        "network κ: {}",
        String::from_utf8_lossy(hologram_space::address_bytes(&bytes).as_array())
    );
    println!("tier: {:?}", net.tier);
    println!("members: {}", net.membership.len());
    for m in &net.membership {
        println!("  {}", String::from_utf8_lossy(m.as_array()));
    }
    println!(
        "policy κ: {}",
        String::from_utf8_lossy(net.policy.as_array())
    );
    match &net.key_ref {
        Some(k) => println!("key κ: {}", String::from_utf8_lossy(k.as_array())),
        None => println!("key κ: none (unencrypted tier)"),
    }
    Ok(())
}

fn run_app(app_cli: AppCli) -> Result<(), CompileError> {
    match app_cli.command {
        AppCommand::Inspect { archive } => app_inspect(&archive),
        AppCommand::Thin { input, output } => app_thin(&input, &output),
        AppCommand::Fat {
            input,
            output,
            store,
        } => app_fat(&input, &output, &store),
    }
}

/// Convert `input` to a **fat** `.holo` at `output`, embedding every layer/closure κ resolvable from
/// the `NativeKappaStore` at `store_dir` (spec 03 §Fat and thin). The app κ is unchanged.
fn app_fat(input: &Path, output: &Path, store_dir: &Path) -> Result<(), CompileError> {
    use hologram_archive::format::SectionKind;
    use hologram_archive::{HoloLoader, HoloWriter};
    use hologram_space::{resolve_closure, KappaStore, REGISTRY};
    use hologram_store::native::NativeKappaStore;

    let bytes = std::fs::read(input).map_err(|_| CompileError::SourceParse("read archive"))?;
    let plan = HoloLoader::from_bytes(&bytes)
        .map_err(CompileError::Archive)?
        .into_plan()
        .map_err(CompileError::Archive)?;
    let manifest = plan
        .app_manifest()
        .ok_or(CompileError::SourceParse(
            "not a .holo v3 application (no manifest section)",
        ))?
        .to_vec();

    let store =
        NativeKappaStore::open(store_dir).map_err(|_| CompileError::SourceParse("open store"))?;
    // Seed the manifest so the closure walk starts from its κ, then resolve over the store.
    let manifest_kappa = store
        .put("blake3", &manifest)
        .map_err(|_| CompileError::SourceParse("store manifest"))?;
    let closure = resolve_closure(manifest_kappa, &store, REGISTRY)
        .map_err(|_| CompileError::SourceParse("resolve closure"))?;

    let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();
    sections.push((SectionKind::AppManifest, manifest));
    if let Ok(certs) = plan.section(SectionKind::Certificates) {
        sections.push((SectionKind::Certificates, certs.to_vec()));
    }
    let mut embedded = 0usize;
    for kappa in &closure.reachable {
        if let Ok(Some(content)) = store.get(kappa) {
            let mut blob = kappa.as_array().to_vec();
            blob.extend_from_slice(content.as_ref());
            sections.push((SectionKind::ContentBlob, blob));
            embedded += 1;
        }
    }
    let fat = HoloWriter::assemble(sections);
    std::fs::write(output, &fat).map_err(|_| CompileError::SourceParse("write archive"))?;
    println!(
        "fattened {} bytes → {} bytes ({embedded} blob(s) embedded) at {}",
        bytes.len(),
        fat.len(),
        output.display()
    );
    Ok(())
}

/// Inspect a `.holo` v3 application: identity κ + layer descriptors (spec 03). Store-free — decodes
/// the manifest realization only.
fn app_inspect(archive: &Path) -> Result<(), CompileError> {
    use hologram_space::AppManifest;
    let bytes = std::fs::read(archive).map_err(|_| CompileError::SourceParse("read archive"))?;
    let plan = hologram_archive::HoloLoader::from_bytes(&bytes)
        .map_err(CompileError::Archive)?
        .into_plan()
        .map_err(CompileError::Archive)?;
    let manifest_bytes = plan.app_manifest().ok_or(CompileError::SourceParse(
        "not a .holo v3 application (no manifest section)",
    ))?;
    let manifest = AppManifest::decode(manifest_bytes)
        .map_err(|_| CompileError::SourceParse("malformed app manifest"))?;
    println!(
        "app κ: {}",
        String::from_utf8_lossy(manifest.kappa().as_array())
    );
    match manifest.primary {
        Some(i) => println!("primary layer: {i}"),
        None => println!("primary layer: none (non-executable / degenerate archive)"),
    }
    println!("layers: {}", manifest.layers.len());
    for (i, l) in manifest.layers.iter().enumerate() {
        let aux = if l.aux.is_empty() {
            String::new()
        } else {
            format!(" aux={:?}", l.aux)
        };
        println!("  [{i}] {:?} entry={:?}{aux}", l.kind, l.entry);
    }
    println!("children: {}", manifest.children.len());
    Ok(())
}

/// Convert `input` to a **thin** `.holo` (manifest + certificates only) at `output`. The app κ is
/// unchanged — fat↔thin is packaging, never identity (spec 03 §Fat and thin).
fn app_thin(input: &Path, output: &Path) -> Result<(), CompileError> {
    let bytes = std::fs::read(input).map_err(|_| CompileError::SourceParse("read archive"))?;
    let thin = thin_archive_bytes(&bytes)?;
    std::fs::write(output, &thin).map_err(|_| CompileError::SourceParse("write archive"))?;
    println!(
        "thinned {} bytes → {} bytes at {}",
        bytes.len(),
        thin.len(),
        output.display()
    );
    Ok(())
}

/// Pure core of `app thin`: keep only the manifest + certificate sections, re-framed (spec 03).
fn thin_archive_bytes(bytes: &[u8]) -> Result<Vec<u8>, CompileError> {
    use hologram_archive::format::SectionKind;
    use hologram_archive::{HoloLoader, HoloWriter};
    let plan = HoloLoader::from_bytes(bytes)
        .map_err(CompileError::Archive)?
        .into_plan()
        .map_err(CompileError::Archive)?;
    let manifest = plan
        .app_manifest()
        .ok_or(CompileError::SourceParse(
            "not a .holo v3 application (no manifest section)",
        ))?
        .to_vec();
    let mut sections: Vec<(SectionKind, Vec<u8>)> = Vec::new();
    sections.push((SectionKind::AppManifest, manifest));
    if let Ok(certs) = plan.section(SectionKind::Certificates) {
        sections.push((SectionKind::Certificates, certs.to_vec()));
    }
    Ok(HoloWriter::assemble(sections))
}

struct CompileArgs {
    backend: String,
    witt_level: u32,
    source: Option<std::path::PathBuf>,
    source_language: Option<String>,
    graph_name: Option<String>,
    output: std::path::PathBuf,
    no_warm: bool,
}

struct SourceCompileArgs {
    source: Option<std::path::PathBuf>,
    source_language: Option<String>,
    graph_name: Option<String>,
    kind: BackendKind,
    witt_level: u32,
}

fn compile_command(args: CompileArgs) -> Result<(), CompileError> {
    let kind = parse_backend(&args.backend)?;
    let out = compile_source(SourceCompileArgs {
        source: args.source,
        source_language: args.source_language,
        graph_name: args.graph_name,
        kind,
        witt_level: args.witt_level,
    })?;
    let archive = maybe_warm_fold(out.archive, args.no_warm)?;
    std::fs::write(&args.output, &archive)
        .map_err(|_| CompileError::SourceParse("write archive"))?;
    print_compile_result(archive.len(), &args.output, &out.stats);
    Ok(())
}

fn compile_source(
    args: SourceCompileArgs,
) -> Result<hologram_compiler::CompilationOutput, CompileError> {
    match args.source.as_deref() {
        Some(path) => compile_source_file(path, &args),
        None => Compiler::new(Graph::new(), args.kind, WittLevel::new(args.witt_level)).compile(),
    }
}

fn compile_source_file(
    path: &Path,
    args: &SourceCompileArgs,
) -> Result<hologram_compiler::CompilationOutput, CompileError> {
    let src =
        std::fs::read_to_string(path).map_err(|_| CompileError::SourceParse("read source"))?;
    let language = source_language_for(path, args.source_language.as_deref())?;
    let options = source_options(args.graph_name.as_deref());
    let program = source::parse_ir_with_options(&src, language, &options)?;
    Compiler::new(
        source::lower_ir(&program)?,
        args.kind,
        WittLevel::new(args.witt_level),
    )
    .compile()
}

fn maybe_warm_fold(archive: Vec<u8>, no_warm: bool) -> Result<Vec<u8>, CompileError> {
    if no_warm {
        return Ok(archive);
    }
    let backend: hologram_compute::CpuBackend<hologram_exec::BufferArena> =
        hologram_compute::CpuBackend::new();
    hologram_exec::fold_archive(&archive, backend)
        .map_err(|_| CompileError::SourceParse("warm fold"))
}

fn print_compile_result(
    archive_len: usize,
    output: &Path,
    stats: &hologram_compiler::CompilationStats,
) {
    println!("compiled {} bytes to {}", archive_len, output.display());
    println!(
        "  nodes={} levels={} validated={} cache_hits={}",
        stats.total_nodes, stats.schedule_levels, stats.validated_units, stats.cache_hits
    );
}

fn parse_backend(s: &str) -> Result<BackendKind, CompileError> {
    match s {
        "cpu" => Ok(BackendKind::Cpu),
        "avx2" => Ok(BackendKind::Avx2),
        "avx512" => Ok(BackendKind::Avx512),
        "neon" => Ok(BackendKind::Neon),
        "metal" => Ok(BackendKind::Metal),
        "wgpu" => Ok(BackendKind::Wgpu),
        // Fail loud: an unrecognized `--backend` must not silently downgrade
        // to CPU (a typo like `--backend wgpi` would otherwise compile for the
        // wrong target without warning).
        _ => Err(CompileError::SourceParse(
            "unknown backend (expected one of: cpu, avx2, avx512, neon, metal, wgpu)",
        )),
    }
}

fn source_language_for(
    path: &Path,
    explicit: Option<&str>,
) -> Result<SourceLanguage, CompileError> {
    source::resolve_source_language(explicit, path_extension(path))
}

fn path_extension(path: &Path) -> Option<&str> {
    path.extension().and_then(|ext| ext.to_str())
}

fn source_options(graph: Option<&str>) -> source::SourceParseOptions {
    match graph {
        Some(graph) => source::SourceParseOptions::new().graph(graph),
        None => source::SourceParseOptions::new(),
    }
}

/// Owned zero-filled input buffers sized to each declared input port — the
/// diagnostic Execute/Bench commands feed dummy zeros, but each port's byte
/// length comes from the archive's declared shape × dtype, not a fixed cap.
fn zero_inputs_for(
    session: &hologram_exec::InferenceSession<
        hologram_compute::CpuBackend<hologram_exec::BufferArena>,
    >,
) -> Vec<Vec<u8>> {
    (0..session.input_count())
        .map(|i| vec![0u8; session.input_byte_len(i)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_source_language_from_extension() {
        assert_eq!(
            source_language_for(Path::new("graph.py"), None).unwrap(),
            SourceLanguage::Python
        );
        assert_eq!(
            source_language_for(Path::new("graph.tsx"), None).unwrap(),
            SourceLanguage::TypeScript
        );
        assert_eq!(
            source_language_for(Path::new("graph.rs"), None).unwrap(),
            SourceLanguage::Rust
        );
        assert_eq!(
            source_language_for(Path::new("graph.txt"), None).unwrap(),
            SourceLanguage::Hologram
        );
    }

    #[test]
    fn explicit_source_language_overrides_extension() {
        assert_eq!(
            source_language_for(Path::new("graph.py"), Some("hologram")).unwrap(),
            SourceLanguage::Hologram
        );
        assert_eq!(
            source_language_for(Path::new("graph.unknown"), Some("ts")).unwrap(),
            SourceLanguage::TypeScript
        );
    }

    #[test]
    fn unknown_source_language_fails_loudly() {
        assert!(source_language_for(Path::new("graph.txt"), Some("ruby")).is_err());
    }

    #[test]
    fn builds_source_options_with_graph_selection() {
        assert_eq!(
            source_options(Some("encoder")).graph_name(),
            Some("encoder")
        );
        assert_eq!(source_options(None).graph_name(), None);
    }

    #[test]
    fn app_thin_keeps_manifest_and_drops_payload() {
        use hologram_archive::format::SectionKind;
        use hologram_archive::{HoloLoader, HoloWriter};
        use hologram_space::{address_bytes, AppManifest, Layer, Realization};

        let manifest = AppManifest {
            primary: Some(0),
            requires: address_bytes(b"caps"),
            layers: vec![Layer::wasm(address_bytes(b"w"), "_start")],
            children: vec![],
        };
        let manifest_bytes = manifest.canonicalize();
        let mut w = HoloWriter::new();
        w.set_app_manifest(manifest_bytes.clone());
        w.add_extension("tokenizer", vec![1, 2, 3]); // a payload section to be dropped
        let fat = w.finish().unwrap();

        let thin = thin_archive_bytes(&fat).unwrap();
        assert!(thin.len() < fat.len(), "thin drops the extension payload");
        let plan = HoloLoader::from_bytes(&thin).unwrap().into_plan().unwrap();
        // The manifest (the app κ) is preserved byte-for-byte; the payload is gone.
        assert_eq!(plan.app_manifest(), Some(manifest_bytes.as_slice()));
        assert!(plan.section(SectionKind::Extension).is_err());
    }

    #[test]
    fn network_create_writes_a_realization_from_content_kappas() {
        use hologram_space::{address_bytes, Network, NetworkTier};

        let dir = std::env::temp_dir().join(format!("holo-net-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let member = dir.join("member.key");
        let policy = dir.join("team.caps");
        let out = dir.join("team.net");
        std::fs::write(&member, b"operator-attestation-key").unwrap();
        std::fs::write(&policy, b"restricted-policy-capset").unwrap();

        net_create(
            std::slice::from_ref(&member),
            &policy,
            "restricted",
            None,
            &out,
        )
        .unwrap();

        // The written file decodes to a Network whose membership/policy are the κs of the inputs.
        let net = Network::decode(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(net.tier, NetworkTier::Restricted);
        assert_eq!(
            net.membership,
            vec![address_bytes(b"operator-attestation-key")]
        );
        assert_eq!(net.policy, address_bytes(b"restricted-policy-capset"));
        assert!(net.key_ref.is_none());

        // The Private tier requires a key; without `--key` it is refused.
        let bad = dir.join("bad.net");
        assert!(net_create(
            std::slice::from_ref(&member),
            &policy,
            "private",
            None,
            &bad
        )
        .is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn app_fat_embeds_store_resolvable_content() {
        use hologram_archive::format::SectionKind;
        use hologram_archive::{HoloLoader, HoloWriter};
        use hologram_space::{AppManifest, KappaStore, Layer, Realization};
        use hologram_store::native::NativeKappaStore;

        let dir = std::env::temp_dir().join(format!("holo-fat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_path = dir.join("store.redb");

        // Provision layer content into a persistent store, then build a thin app over those κs. The
        // store is dropped at the end of this block so `app_fat` can re-open the (locked) redb file.
        let manifest = {
            let store = NativeKappaStore::open(&store_path).unwrap();
            AppManifest {
                primary: Some(0),
                requires: store.put("blake3", b"fat-cli-caps").unwrap(),
                layers: vec![
                    Layer::wasm(store.put("blake3", b"fat-cli-wasm").unwrap(), "_start"),
                    Layer::tensor(store.put("blake3", b"fat-cli-plan").unwrap(), "s"),
                ],
                children: vec![],
            }
        };
        let mut w = HoloWriter::new();
        w.set_app_manifest(manifest.canonicalize());
        let thin = dir.join("app.holo");
        std::fs::write(&thin, w.finish().unwrap()).unwrap();

        // Fatten it against the store.
        let fat = dir.join("app.fat.holo");
        app_fat(&thin, &fat, &store_path).unwrap();

        let fat_bytes = std::fs::read(&fat).unwrap();
        let fat_plan = HoloLoader::from_bytes(&fat_bytes)
            .unwrap()
            .into_plan()
            .unwrap();
        // Content is embedded as blobs, and the app κ (manifest bytes) is unchanged.
        assert!(fat_plan.content_blobs().unwrap().len() >= 3);
        assert_eq!(
            fat_plan.app_manifest(),
            Some(manifest.canonicalize().as_slice())
        );
        assert!(fat_plan.section(SectionKind::ContentBlob).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn network_delegate_enforces_attenuation() {
        use hologram_space::{address_bytes, Capabilities, CapabilitySet, Delegation, Realization};

        let caps = |roots: &[&[u8]], quota: u64| Capabilities {
            storage_roots: roots.iter().map(|r| address_bytes(r)).collect(),
            storage_quota_bytes: quota,
            network_fetch: true,
            network_announce: false,
            publish_channels: vec![],
            subscribe_channels: vec![],
            memory_max_bytes: quota,
            cpu_time_per_event_ms: 10,
            priority_weight: 4,
        };

        let dir = std::env::temp_dir().join(format!("holo-deleg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let parent = dir.join("parent.caps");
        let child = dir.join("child.caps");
        let amp = dir.join("amp.caps");
        let out = dir.join("grant.deleg");
        let write = |p: &std::path::Path, c: Capabilities| {
            std::fs::write(p, CapabilitySet::new(c).canonicalize()).unwrap()
        };
        write(&parent, caps(&[b"A", b"B"], 1000));
        write(&child, caps(&[b"A"], 500)); // subset → attenuated
        write(&amp, caps(&[b"A", b"C"], 500)); // reaches root C the parent lacks → amplifies

        // The attenuated child is delegated: a valid Delegation binding the two capset κs.
        net_delegate(&parent, &child, &out).unwrap();
        assert_eq!(
            Delegation::references(&std::fs::read(&out).unwrap())
                .unwrap()
                .len(),
            2
        );
        // The amplifying child is refused (attenuation only, law 5).
        assert!(net_delegate(&parent, &amp, &out).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
