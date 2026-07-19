# hologram-bench

> Criterion benchmarks for the Hologram runtime.

The workspace's performance battery. The crate's library carries no benchmarks — only the
Criterion `[[bench]]` targets in `benches/` — and the lib's default libtest bench harness is
disabled so `cargo bench` can forward Criterion arguments (e.g. `--measurement-time`) without
libtest rejecting them. The benches exercise the compiler, executor, compute, ops, graph, and
archive crates across the stack.

## What it provides

Criterion bench targets (run with `cargo bench -p hologram-bench`):

- `kernel_perf`, `matmul`, `decode_gemv` — core kernel and matmul throughput.
- `decode_step`, `tiered_executor`, `production` — decode-step and executor tiers.
- `compiler`, `source_lowering`, `fusion` — compilation, lowering, and fusion passes.
- `lut_activation`, `dequant_activation` — activation LUT and dequant paths.
- `content_reuse` — content-addressed reuse across the substrate.

## Features

- `parallel` — engages the backend's worker pool (`hologram-compute` / `hologram-exec` `parallel`) so the benches measure the UOR hierarchical model leveraging the whole processor surface (`cargo bench -p hologram-bench --features parallel`).

Part of the [hologram](../../README.md) workspace.
