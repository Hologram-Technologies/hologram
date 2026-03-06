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

# Install git hooks
hooks:
    git config core.hooksPath .githooks
    chmod +x .githooks/pre-commit
