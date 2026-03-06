#[tokio::main]
async fn main() {
    if let Err(e) = holo_cli::run().await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
