# Hologram — build commands

set dotenv-load := true

# Default recipe: list all available recipes
default:
    @just --list

# Full CI: format check, clippy, tests
ci: fmt-check clippy test

# Run all tests
test:
    cargo test --workspace

# Run criterion benchmarks
bench:
    cargo bench --workspace

# Format all code
fmt:
    cargo fmt --all

# Check formatting (no changes)
fmt-check:
    cargo fmt --all -- --check

# Clippy with deny warnings
clippy:
    cargo clippy --workspace -- -D warnings

# Build hologram-core for WASM target (std)
wasm:
    cargo build --target wasm32-unknown-unknown -p hologram-core --no-default-features

# Build hologram-core for WASM with no_std + no rkyv (constrained device validation)
wasm-nostd:
    RUSTC=~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc \
    ~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo build \
    --target wasm32-unknown-unknown -p hologram-core --no-default-features

# Build hologram-core for ARM bare-metal (thumbv7em, no_std)
embedded:
    RUSTC=~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc \
    ~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo build \
    --target thumbv7em-none-eabihf -p hologram-core --no-default-features

# Build all
build:
    cargo build --workspace

# Clean
clean:
    cargo clean

# Install site dependencies
site-install:
    cd site && pnpm install

# Start the docs site dev server
site-dev:
    cd site && pnpm dev

# Build the docs site
site-build:
    cd site && pnpm build

# Preview the built docs site
site-preview:
    cd site && pnpm preview

# Deploy the docs site (via GitHub Pages — push to main triggers the workflow)
site-deploy: site-build
    @echo "Site deploys automatically via GitHub Actions on push to main."
    @echo "To trigger manually: gh workflow run deploy-site.yml"

# Install required dependencies (Rust toolchain assumed)
install:
    cd site && pnpm install

# Install optional dependencies (WASM tooling, benchmark visualization)
install-optional:
    cargo install wasm-pack wasm-bindgen-cli
    rustup target add wasm32-unknown-unknown

# Build WASM module for the demo page (uses rustup toolchain to find wasm32 target)
wasm-demo:
    RUSTUP_TOOLCHAIN=stable RUSTC="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc" "$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo" build --target wasm32-unknown-unknown -p hologram-ffi --features wasm --no-default-features --release
    wasm-bindgen target/wasm32-unknown-unknown/release/hologram_ffi.wasm --out-dir site/public/demo/pkg --target web

# Build WASM and start the calculator demo
demo: wasm-demo
    cd site && pnpm dev

# Install git hooks
hooks:
    git config core.hooksPath .githooks
    chmod +x .githooks/pre-commit
