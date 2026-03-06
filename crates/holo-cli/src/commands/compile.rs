//! `hologram compile` — compile source to `.holo` archive.

use crate::error::CliError;
use clap::Args;
use std::path::PathBuf;

/// Arguments for the compile subcommand.
#[derive(Args)]
pub struct CompileArgs {
    /// Input source file.
    pub input: PathBuf,
    /// Output `.holo` file path.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Execute the compile command.
pub async fn execute(args: CompileArgs) -> Result<(), CliError> {
    let output = args
        .output
        .unwrap_or_else(|| args.input.with_extension("holo"));
    println!("Compiling {:?} -> {:?}", args.input, output);
    Ok(())
}
