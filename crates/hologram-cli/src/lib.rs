//! Async CLI with subcommands for hologram operations.
//!
//! Exposes a single `run()` function for the binary entry point.

pub mod commands;
pub mod error;
pub mod fmt;

use clap::Parser;
use commands::Command;

/// Hologram — O(1) compute acceleration via pre-computed lookup tables.
#[derive(Parser)]
#[command(name = "hologram", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Run the CLI. Call this from `main()`.
pub async fn run() -> Result<(), error::CliError> {
    let cli = Cli::parse();
    commands::dispatch(cli.command).await
}
