//! CLI subcommands.

pub mod compile;
pub mod inspect;
pub mod run_cmd;

use crate::error::CliError;
use clap::Subcommand;

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Compile a source file to `.holo` archive format.
    Compile(compile::CompileArgs),
    /// Execute a `.holo` file with provided inputs.
    Run(run_cmd::RunArgs),
    /// Inspect a `.holo` archive: print metadata without executing.
    Inspect(inspect::InspectArgs),
}

/// Dispatch a parsed command to its handler.
pub async fn dispatch(cmd: Command) -> Result<(), CliError> {
    match cmd {
        Command::Compile(args) => compile::execute(args).await,
        Command::Run(args) => run_cmd::execute(args).await,
        Command::Inspect(args) => inspect::execute(args).await,
    }
}
