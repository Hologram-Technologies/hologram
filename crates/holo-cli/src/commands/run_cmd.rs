//! `hologram run` — execute a `.holo` file.

use crate::error::CliError;
use clap::Args;
use std::path::PathBuf;

/// Arguments for the run subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
}

/// Execute the run command.
pub async fn execute(args: RunArgs) -> Result<(), CliError> {
    println!("Running {:?}", args.file);
    Ok(())
}
