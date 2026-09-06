//! `hologram ai` — AI application tooling over `.holo` v4 inference-model layers
//! (spec `refactor/03` §v4).
//!
//! **Feature `ai` (default off).** This module is a thin adapter: clap parsing →
//! `hologram_ai::cli` → print. Hologram itself stays engine-agnostic — every AI
//! concern (model download, model compile, the engine runtime, inference) lives in
//! the sibling `hologram-ai` crate (hologram-ai repository). Hologram never depends
//! on an engine in its default build.
//!
//! # Wiring (hologram-ai is not yet published)
//!
//! The dependency is commented out in `crates/hologram-cli/Cargo.toml`: the
//! `hologram-ai` crate does not exist on crates.io yet, and an optional *path*
//! dependency on a non-existent path breaks `cargo check --workspace` resolution
//! even with the feature off. To build with `--features ai`:
//!
//! 1. Uncomment the `hologram-ai` path dependency in `Cargo.toml` (adjust the path
//!    to your hologram-ai checkout; it becomes a git pin once hologram-ai is
//!    published/pinned).
//! 2. Change the feature to `ai = ["dep:hologram-ai"]`.
//!
//! Enabling `ai` without that dependency is a deliberate compile error (unresolved
//! `hologram_ai` import) — there is no stub fallback.
//!
//! # Assumed `hologram-ai` API (the contract this adapter is written against)
//!
//! ```ignore
//! // hologram_ai::cli — one entry point per `hologram ai` verb. `CliOutput.text`
//! // is human-readable text, or a JSON document when the verb's `--json` was set.
//! pub struct CliOutput { pub code: i32, pub text: String }
//! pub struct DownloadArgs { pub repository: String, pub revision: String, pub offline: bool, pub json: bool }
//! pub struct CompileArgs {
//!     pub repository: Option<String>, pub revision: Option<String>,
//!     pub source: Option<std::path::PathBuf>, pub output: std::path::PathBuf,
//!     pub entry: String, pub json: bool,
//! }
//! pub struct InspectArgs { pub file: std::path::PathBuf, pub json: bool }
//! pub struct InferArgs {
//!     pub file: std::path::PathBuf, pub model: Option<String>, pub operation: String,
//!     pub prompt: Option<String>, pub max_output_tokens: Option<u32>, pub json: bool,
//! }
//! pub fn download(args: DownloadArgs) -> CliOutput;
//! pub fn compile(args: CompileArgs) -> CliOutput;
//! pub fn inspect(args: InspectArgs) -> CliOutput;
//! pub fn infer(args: InferArgs) -> CliOutput;
//! ```

use clap::Subcommand;
use hologram_compiler::error::CompileError;
use std::path::PathBuf;

/// `hologram ai <subcommand>` — AI application tooling (delegates to `hologram-ai`).
#[derive(clap::Args, Debug)]
pub struct AiCli {
    #[command(subcommand)]
    command: AiCommand,
}

#[derive(Subcommand, Debug)]
enum AiCommand {
    /// Download a pinned model repository revision into the local cache.
    Download(DownloadArgs),
    /// Compile a model into a `.holo` v4 application with an inference-model layer.
    Compile(CompileArgs),
    /// Inspect a `.holo` AI application: its inference-model layers + engine tags.
    Inspect(InspectArgs),
    /// Invoke an inference-model layer's service in a `.holo` AI application.
    Infer(InferArgs),
}

/// `hologram ai download <repository> --revision <full-sha> [--offline] [--json]`.
#[derive(clap::Args, Debug)]
struct DownloadArgs {
    /// Model repository (e.g. a Hugging Face repo id).
    repository: String,
    /// Full commit SHA to pin — κ-only identity: a floating ref is not an identity.
    #[arg(long)]
    revision: String,
    /// Resolve from the local cache only — fail rather than touch the network.
    #[arg(long)]
    offline: bool,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

/// `hologram ai compile [repository] [--revision <sha> | --source <dir>] --output
/// <file.holo> [--entry <name>] [--json]`.
#[derive(clap::Args, Debug)]
struct CompileArgs {
    /// Model repository (omit when compiling from a local `--source` tree).
    repository: Option<String>,
    /// Full commit SHA to pin (with `repository`).
    #[arg(long, conflicts_with = "source")]
    revision: Option<String>,
    /// Local source directory to compile instead of a repository.
    #[arg(long)]
    source: Option<PathBuf>,
    /// Output `.holo` application path.
    #[arg(long)]
    output: PathBuf,
    /// Callable service name for the inference-model layer's entry.
    #[arg(long, default_value = "ai.default")]
    entry: String,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

/// `hologram ai inspect <file.holo> [--json]`.
#[derive(clap::Args, Debug)]
struct InspectArgs {
    /// The `.holo` application to inspect.
    file: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

/// `hologram ai infer <file.holo> [--model <entry>] [--operation <op>] [--prompt
/// <text>] [--max-output-tokens <n>] [--json]`.
#[derive(clap::Args, Debug)]
struct InferArgs {
    /// The `.holo` application to invoke.
    file: PathBuf,
    /// Service entry to invoke (defaults to the app's default model).
    #[arg(long)]
    model: Option<String>,
    /// Operation to invoke on the service.
    #[arg(long, default_value = "generate")]
    operation: String,
    /// Prompt text for the operation.
    #[arg(long)]
    prompt: Option<String>,
    /// Cap on generated output tokens.
    #[arg(long)]
    max_output_tokens: Option<u32>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

/// Run `hologram ai …`: parse → delegate to `hologram_ai::cli` → print the payload.
pub fn run(ai_cli: AiCli) -> Result<(), CompileError> {
    finish(ai_cli.command.dispatch())
}

impl AiCommand {
    fn dispatch(self) -> hologram_ai::cli::CliOutput {
        match self {
            AiCommand::Download(a) => hologram_ai::cli::download(a.into()),
            AiCommand::Compile(a) => hologram_ai::cli::compile(a.into()),
            AiCommand::Inspect(a) => hologram_ai::cli::inspect(a.into()),
            AiCommand::Infer(a) => hologram_ai::cli::infer(a.into()),
        }
    }
}

fn finish(output: hologram_ai::cli::CliOutput) -> Result<(), CompileError> {
    println!("{}", output.text);
    if output.code == 0 {
        Ok(())
    } else {
        Err(CompileError::SourceParse("ai command failed"))
    }
}

impl From<DownloadArgs> for hologram_ai::cli::DownloadArgs {
    fn from(a: DownloadArgs) -> Self {
        Self {
            repository: a.repository,
            revision: a.revision,
            offline: a.offline,
            json: a.json,
        }
    }
}

impl From<CompileArgs> for hologram_ai::cli::CompileArgs {
    fn from(a: CompileArgs) -> Self {
        Self {
            repository: a.repository,
            revision: a.revision,
            source: a.source,
            output: a.output,
            entry: a.entry,
            json: a.json,
        }
    }
}

impl From<InspectArgs> for hologram_ai::cli::InspectArgs {
    fn from(a: InspectArgs) -> Self {
        Self {
            file: a.file,
            json: a.json,
        }
    }
}

impl From<InferArgs> for hologram_ai::cli::InferArgs {
    fn from(a: InferArgs) -> Self {
        Self {
            file: a.file,
            model: a.model,
            operation: a.operation,
            prompt: a.prompt,
            max_output_tokens: a.max_output_tokens,
            json: a.json,
        }
    }
}
