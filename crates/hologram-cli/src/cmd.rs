//! CLI subcommands.

use clap::{Parser, Subcommand};
use hologram_compiler::{Compiler, BackendKind};
use hologram_compiler::error::CompileError;
use hologram_graph::Graph;
use uor_foundation::WittLevel;

#[derive(Parser, Debug)]
#[command(name = "hologram", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
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
}

pub fn run(cli: Cli) -> Result<(), CompileError> {
    match cli.command {
        Command::Compile { backend, witt_level, source, output } => {
            let kind = parse_backend(&backend);
            let level = WittLevel::new(witt_level);
            let graph = match source {
                Some(path) => {
                    let src = std::fs::read_to_string(&path)
                        .map_err(|_| CompileError::SourceParse("read source"))?;
                    hologram_compiler::source::parse(&src)?
                }
                None => Graph::new(),
            };
            let out = Compiler::new(graph, kind, level).compile()?;
            std::fs::write(&output, &out.archive)
                .map_err(|_| CompileError::SourceParse("write archive"))?;
            println!("compiled {} bytes to {}", out.archive.len(), output.display());
            println!(
                "  nodes={} levels={} validated={} cache_hits={}",
                out.stats.total_nodes,
                out.stats.schedule_levels,
                out.stats.validated_units,
                out.stats.cache_hits,
            );
            Ok(())
        }
        Command::Execute { archive } => {
            let bytes = std::fs::read(&archive)
                .map_err(|_| CompileError::SourceParse("read archive"))?;
            let backend: hologram_backend::CpuBackend<hologram_exec::BufferArena> =
                hologram_backend::CpuBackend::new();
            let mut session = hologram_exec::InferenceSession::load(&bytes, backend)
                .map_err(|_| CompileError::SourceParse("load archive"))?;
            let zeros = vec![0u8; 4096];
            let inputs: Vec<hologram_exec::InputBuffer> = (0..session.input_count())
                .map(|_| hologram_exec::InputBuffer { bytes: &zeros })
                .collect();
            let outputs = session.execute(&inputs)
                .map_err(|_| CompileError::SourceParse("execute"))?;
            for (i, o) in outputs.iter().enumerate() {
                println!("output[{i}] = {} bytes", o.bytes.len());
            }
            Ok(())
        }
        Command::Inspect { archive } => {
            let bytes = std::fs::read(&archive)
                .map_err(|_| CompileError::SourceParse("read archive"))?;
            let plan = hologram_archive::HoloLoader::from_bytes(&bytes)
                .map_err(CompileError::Archive)?
                .into_plan()
                .map_err(CompileError::Archive)?;
            for s in plan.sections() {
                println!("section {:?} @ {} ({} bytes)", s.kind, s.offset, s.length);
            }
            Ok(())
        }
    }
}

fn parse_backend(s: &str) -> BackendKind {
    match s {
        "avx2" => BackendKind::Avx2,
        "avx512" => BackendKind::Avx512,
        "neon" => BackendKind::Neon,
        "metal" => BackendKind::Metal,
        "wgpu" => BackendKind::Wgpu,
        _ => BackendKind::Cpu,
    }
}
