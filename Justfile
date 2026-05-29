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
vv: fmt-check clippy test conformance parallel perf wasm embedded
    @echo "V&V complete — see CONFORMANCE.md for the invariant catalog."

# External-authority + scaling conformance suites (classes AS/MA/KC/SC).
conformance:
    cargo test -p hologram-archive --test conformance --test model_address --features model-formats
    cargo test -p hologram-backend --test conformance --features cpu
    cargo test -p hologram-exec --test conformance

# Parallel-execution conformance (class PA): multi-core ≡ single-thread,
# byte-identical + deterministic. Runs the kernel suites with the in-tree
# worker pool active so the parallel lattice-recursion frontier is exercised.
parallel:
    cargo test -p hologram-backend --features cpu,parallel --test parallel --test conformance --lib cpu::parallel

# Performance V&V (class PV) — release-only budgets; no silent bottleneck.
# `--nocapture` surfaces PV-4's production throughput / FLOP-per-core-cycle report.
# Also runs the deployment-substrate SP-class criterion floors (G1/G2 native store, mem zero-copy).
perf:
    cargo test --release -p hologram-backend --test performance --features cpu -- --nocapture
    cargo test --release -p hologram-exec --test performance -- --nocapture
    cargo bench -p hologram-store-native --bench sp_floors -- --quick
    cargo bench -p hologram-store-mem --bench sp_floors -- --quick

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
    # Deployment substrate (TR class): the portable reference + runtime build no_std for the browser.
    cargo build --target wasm32-unknown-unknown --no-default-features \
        -p hologram-substrate-core -p hologram-realizations -p hologram-store-mem \
        -p hologram-net-http -p hologram-runtime \
        -p hologram-bare-hal -p hologram-net-bare -p hologram-runtime-bare

# Build the no_std library stack for bare-metal ARM (thumbv7em, no std sysroot).
embedded:
    cargo build --target thumbv7em-none-eabi --no-default-features \
        -p hologram-host -p hologram-types -p hologram-ops -p hologram-graph \
        -p hologram-archive -p hologram-compiler -p hologram-exec
    cargo build --target thumbv7em-none-eabi --no-default-features --features cpu \
        -p hologram-backend
    # Deployment substrate (TR class): same source builds no_std for the bare-metal substrate.
    cargo build --target thumbv7em-none-eabi --no-default-features \
        -p hologram-substrate-core -p hologram-realizations -p hologram-store-mem \
        -p hologram-net-http -p hologram-runtime -p hologram-bare-hal -p hologram-store-bare \
        -p hologram-net-bare

# Deployment-substrate V&V (see specs/docs/container-substrate-vv.md): conformance + worked example
# + SP floors across native, then the no_std tripling builds. RZ gate: the tensor compute engine
# (hologram-exec/-backend) must NOT appear in the store/route crates' dependency tree.
vv-substrate:
    cargo test -p hologram-substrate-core -p hologram-realizations -p hologram-substrate-tck \
        -p hologram-store-mem -p hologram-store-native -p hologram-net-http -p hologram-net-tcp \
        -p hologram-runtime -p hologram-substrate-cli -p hologram-runtime-wasmtime \
        -p hologram-bare-hal -p hologram-store-bare -p hologram-runtime-bare -p hologram-net-bare
    cargo test -p hologram-net-http --features live   # live HTTP-CAS transport
    @echo "RZ gate — compute engine (exec/backend/ops/graph/compiler/archive) absent from store/route:"
    @for c in hologram-store-mem hologram-store-native hologram-store-bare hologram-net-http hologram-runtime hologram-runtime-wasmtime hologram-runtime-bare hologram-net-bare hologram-substrate-cli; do \
        cargo tree -p $c -e normal 2>/dev/null | grep -E "hologram-(exec|backend|ops|graph|compiler|archive)" \
        && (echo "RZ VIOLATION in $c" && exit 1) || echo "  $c: RZ ok"; \
    done
    just wasm embedded

# End-to-end bare-metal boot: build hologram.efi and boot it in QEMU/OVMF (no OS), asserting the
# engine's storage self-check prints PASS. Requires qemu-system-x86_64 + OVMF + x86_64-unknown-uefi.
uefi-boot:
    ./scripts/uefi-boot-test.sh

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

# End-to-end OPFS browser-store test: build the wasm32 store + run it in Chromium via Playwright
# (put/get round-trip, reload persistence, verify-on-receipt). Requires wasm-bindgen + node + Playwright.
opfs-test:
    ./scripts/opfs-browser-test.sh

# Run the real-world container examples (CAS cache, event bus, least-privilege, Wasm inference,
# live migration) — each a runnable narrative of a substrate capability.
examples:
    cargo run -q -p hologram-runtime-wasmtime --example cas_artifact_cache
    cargo run -q -p hologram-runtime-wasmtime --example event_bus
    cargo run -q -p hologram-runtime-wasmtime --example least_privilege
    cargo run -q -p hologram-runtime-wasmtime --example wasm_inference_container
    cargo run -q -p hologram-runtime-wasmtime --example live_migration
