# Hologram — build commands

set dotenv-load := true

# Default recipe: list all available recipes
default:
    @just --list

# Full CI: format check, clippy, tests, supply-chain gate
ci: fmt-check clippy test deny

# Supply-chain gate (class GV): licenses stay in the permissive set (MIT/Apache + the
# reviewed exceptions), no vulnerable/unsound/yanked crates, sources are crates.io. This
# is the RUSTSEC advisory audit *and* the license/ban/source policy in one. See deny.toml.
deny:
    cargo deny check

# Verification & Validation (see VERIFICATION.md / CONFORMANCE.md).
# Every part validated against an external authority + portability +
# performance. Conformance suites are the `*::conformance` test targets.
vv: fmt-check clippy test conformance bdd parallel perf deny wasm embedded
    @echo "V&V complete — see CONFORMANCE.md for the invariant catalog."

# External-authority + scaling conformance suites (classes AS/MA/KC/SC).
conformance:
    cargo test -p hologram-archive --test conformance --test model_address --features model-formats
    cargo test -p hologram-compute --test conformance --features cpu
    cargo test -p hologram-exec --test conformance

# BDD conformance suite (refactor classes LAW/SP/HF/NW/TL/MG/GV). Runs the cucumber
# runner + the honesty meta-gate (catalog ↔ scenario bijection). See features/README.md.
bdd:
    cargo test -p hologram-conformance --test bdd
    cargo test -p hologram-conformance --test meta_gate

# Verify the BDD status column against actual scenario tags (static check). Fails on drift.
conformance-report:
    cargo test -p hologram-conformance --test meta_gate

# Parallel-execution conformance (class PA): multi-core ≡ single-thread,
# byte-identical + deterministic. Runs the kernel suites with the in-tree
# worker pool active so the parallel lattice-recursion frontier is exercised.
parallel:
    cargo test -p hologram-compute --features cpu,parallel --test parallel --test conformance --lib cpu::parallel

# Performance V&V (class PV) — release-only budgets; no silent bottleneck.
# `--nocapture` surfaces PV-4's production throughput / FLOP-per-core-cycle report.
# Also runs the deployment-substrate SP-class criterion floors (G1/G2 native store, mem zero-copy).
perf:
    cargo test --release -p hologram-compute --test performance --features cpu -- --nocapture
    cargo test --release -p hologram-exec --test performance -- --nocapture
    cargo bench -p hologram-store --features native --bench sp_floors -- --quick
    cargo bench -p hologram-tck --bench sp_floors -- --quick

# Run all tests. nextest skips the cucumber `bdd` runner (harness=false — see
# .config/nextest.toml), so run that suite explicitly with cargo test afterward.
test:
    cargo nextest run --workspace
    cargo test -p hologram-conformance --test bdd

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
# `#![no_std]` path is exercised; `hologram-compute` adds its CPU kernels.
wasm:
    cargo build --target wasm32-unknown-unknown --no-default-features \
        -p hologram-types -p hologram-ops -p hologram-graph \
        -p hologram-archive -p hologram-compiler -p hologram-exec
    # The space contract builds no_std for wasm32 — the wasm half of the SP-3 composition proof
    # wasm32 — the wasm half of the composition proof (native half is the SP-3 BDD run).
    cargo build --target wasm32-unknown-unknown --no-default-features \
        -p hologram-space
    cargo build --target wasm32-unknown-unknown --no-default-features --features cpu \
        -p hologram-compute
    # SIMD tiers: baseline simd128 (the witnessed browser kernel) and the
    # relaxed-SIMD tier (i8 relaxed dot — same exact function, engine-fast
    # path). Building both keeps the cfg'd kernels from bit-rotting.
    RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown \
        --no-default-features --features cpu -p hologram-compute
    RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo build --target wasm32-unknown-unknown \
        --no-default-features --features cpu -p hologram-compute
    # Embedder-worker pool (shared-memory build; imports the embedder futex).
    RUSTFLAGS="-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals" \
        cargo build --target wasm32-unknown-unknown \
        --no-default-features --features cpu,wasm-threads -p hologram-compute
    # Threaded lane (wasip1-threads carries atomics in its target spec); the
    # fork-join bit-identity test runs under `wasmtime -W threads=y -S threads`.
    RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-wasip1-threads \
        --no-default-features --features cpu,std,wasm-threads -p hologram-compute
    # Deployment substrate (TR class): the consolidated space/tck/net/runtime crates build
    # no_std for the browser, and the wasmi engine (browser/iOS interpreter) builds for wasm32.
    cargo build --target wasm32-unknown-unknown --no-default-features \
        -p hologram-space -p hologram-tck -p hologram-net -p hologram-runtime
    cargo build --target wasm32-unknown-unknown --no-default-features --features engine-wasmi \
        -p hologram-runtime

# Build the no_std library stack for bare-metal ARM (thumbv7em, no std sysroot).
embedded:
    cargo build --target thumbv7em-none-eabi --no-default-features \
        -p hologram-types -p hologram-ops -p hologram-graph \
        -p hologram-archive -p hologram-compiler -p hologram-exec
    cargo build --target thumbv7em-none-eabi --no-default-features --features cpu \
        -p hologram-compute
    # Deployment substrate (TR class): same source builds no_std for the bare-metal substrate.
    cargo build --target thumbv7em-none-eabi --no-default-features \
        -p hologram-space -p hologram-tck -p hologram-net -p hologram-runtime
    cargo build --target thumbv7em-none-eabi --no-default-features --features bare -p hologram-store

# Deployment-substrate V&V (see specs/docs/container-substrate-vv.md): conformance + worked example
# + SP floors across native, then the no_std tripling builds. RZ gate: the tensor compute engine
# (hologram-exec/-compute) must NOT appear in the store/route crates' dependency tree.
vv-substrate:
    cargo test -p hologram-space -p hologram-tck \
        -p hologram-net -p hologram-runtime
    cargo test -p hologram-store --features bare,native   # the merged store's backend TCK tests
    cargo test -p hologram-net --features live,tcp        # live HTTP-CAS + κ-XOR DHT transports
    cargo test -p hologram-runtime --features engine-wasmtime   # the Wasmtime engine backend
    @echo "RZ gate — compute engine (exec/compute/ops/graph/compiler/archive) absent from store/route:"
    @for c in hologram-tck hologram-store hologram-net hologram-runtime; do \
        cargo tree -p $c -e normal 2>/dev/null | grep -E "hologram-(exec|compute|ops|graph|compiler|archive)" \
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
    rustup target add wasm32-unknown-unknown wasm32-wasip1 wasm32-wasip1-threads
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
    cargo run -q -p hologram-runtime --features engine-wasmtime --example cas_artifact_cache
    cargo run -q -p hologram-runtime --features engine-wasmtime --example event_bus
    cargo run -q -p hologram-runtime --features engine-wasmtime --example least_privilege
    cargo run -q -p hologram-runtime --features engine-wasmtime --example wasm_inference_container
    cargo run -q -p hologram-runtime --features engine-wasmtime --example live_migration

# ── Release ──────────────────────────────────────────────────────────────────
# Cutting a release IS the `version-bump` GitHub workflow: it bumps every crate + SDK to the new
# version, regenerates the driver lockfiles, snapshots the public API, updates the changelog,
# commits, tags `vX.Y.Z`, and pushes. The tag then triggers `publish.yml` (crates.io/npm/PyPI) and
# the release-tier CI in `release.yml` (heavy conformance + SDK packaging + perf gate vs the previous
# tag). These recipes just dispatch that one tested workflow (needs `gh` auth) — nothing runs locally,
# so your working tree/branch is irrelevant (the workflow always releases from `main`).

# Cut a release at an EXPLICIT version — e.g. `just release 0.12.0` (or `just release 0.12.0-rc.1`).
release version:
    @echo "{{version}}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' || { echo "✗ '{{version}}' is not a semver (X.Y.Z or X.Y.Z-pre)"; exit 1; }
    gh workflow run version-bump.yml -f version="{{version}}"
    @echo "→ release v{{version}} dispatched. Watch: gh run watch  (or the Actions tab)."

# Cut a release by AUTO-bumping from the current version.
#   just release-auto            → patch        just release-auto minor
#   just release-auto major                     just release-auto minor rc   (pre-release)
release-auto bump="patch" pre="":
    gh workflow run version-bump.yml -f version_type="{{bump}}" -f prerelease="{{pre}}"
    @echo "→ {{bump}} release (prerelease='{{pre}}') dispatched. Watch: gh run watch  (or the Actions tab)."
