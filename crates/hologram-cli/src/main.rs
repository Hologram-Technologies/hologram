fn main() {
    if let Err(e) = hologram_cli::cmd::run_from_env() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
