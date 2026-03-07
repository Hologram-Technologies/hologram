#[tokio::main]
async fn main() {
    if let Err(e) = hologram_cli::run().await {
        let code = e.exit_code();
        eprintln!("Error: {e}");
        std::process::exit(code);
    }
}
