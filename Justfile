# Hologram — build commands

set dotenv-load := true

# Default recipe: list all available recipes
default:
    @just --list

# Full CI: format check, clippy, tests
ci: fmt-check clippy test

# Verification & Validation (see VERIFICATION.md / CONFORMANCE.md).
# Every part validated against an external authority + portability +
# performance. Conformance suites are the `*::conformance` test targets.
vv: fmt-check clippy test conformance perf wasm embedded
    @echo "V&V complete — see CONFORMANCE.md for the invariant catalog."

# External-authority + scaling conformance suites (classes AS/MA/KC/SC).
conformance:
    cargo test -p hologram-archive --test conformance --test model_address --features model-formats
    cargo test -p hologram-backend --test conformance --features cpu
    cargo test -p hologram-exec --test conformance

# Performance V&V (class PV) — release-only budgets; no silent bottleneck.
# `--nocapture` surfaces PV-4's production throughput / FLOP-per-core-cycle report.
perf:
    cargo test --release -p hologram-backend --test performance --features cpu -- --nocapture
    cargo test --release -p hologram-exec --test performance -- --nocapture

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

# Build the no_std library stack for wasm32 (hologram-ai's deploy target).
# `--no-default-features` deactivates every crate's `std` feature, so the
# `#![no_std]` path is exercised; `hologram-backend` adds its CPU kernels.
wasm:
    cargo build --target wasm32-unknown-unknown --no-default-features \
        -p hologram-host -p hologram-types -p hologram-ops -p hologram-graph \
        -p hologram-archive -p hologram-compiler -p hologram-exec
    cargo build --target wasm32-unknown-unknown --no-default-features --features cpu \
        -p hologram-backend

# Build the no_std library stack for bare-metal ARM (thumbv7em, no std sysroot).
embedded:
    cargo build --target thumbv7em-none-eabi --no-default-features \
        -p hologram-host -p hologram-types -p hologram-ops -p hologram-graph \
        -p hologram-archive -p hologram-compiler -p hologram-exec
    cargo build --target thumbv7em-none-eabi --no-default-features --features cpu \
        -p hologram-backend

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

# Install optional dependencies (WASM tooling, benchmark visualization, deployment)
install-optional:
    cargo install wasm-pack wasm-bindgen-cli
    rustup target add wasm32-unknown-unknown
    npm install -g wrangler

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
