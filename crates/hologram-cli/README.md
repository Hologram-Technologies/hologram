# hologram-cli

> The one `hologram` binary — compile, execute, bench, and inspect `.holo` archives, plus the deployment-substrate node.

This crate builds the `hologram` command-line tool. It is a host binary: it opts the otherwise-`no_std` library crates back into `std` and selects the CPU backend, consuming the compiler, executor, and archive stack through the `uor-prism` façade (ADR-031). The former `hologram-substrate-cli` is folded in as the `node` subcommand group (D13).

## What it provides

The `hologram` binary (`src/main.rs` → `hologram_cli::cmd::run_from_env`) with these subcommands:

- `compile` — compile a hologram source file (or an empty graph) to a `.holo` archive, selecting `--backend`, `--witt-level`, and `--source-language`; by default it materializes the warm-start fold (WS-2), suppressed with `--no-warm`.
- `execute` — run a `.holo` archive against the CPU backend with zero-byte inputs, reporting the byte length of each declared output port.
- `inspect` — dump a `.holo` archive's section table plus kernel-call and exec-plan structure.
- `bench` — run an archive `--iterations` times against zero inputs and report wall-clock per iteration.
- `node` — the deployment-substrate node (`hologram node …`): a native redb `NativeKappaStore` with `put`/`get`/`pin`/`unpin`/`gc` verbs (Wasmtime engine + uor-native TCP transports back the forthcoming `spawn`/`serve` verbs).
- `app` — `.holo` v3 application tooling (spec `refactor/03`): `inspect` a manifest's layers/certificates, or convert between `fat` (embed store-resolvable content) and `thin` (manifest + certificates only) without changing the app κ.
- `network` — Network (VPC-analogue) tooling (spec `refactor/04`): `create` and `show` a κ-addressed Network realization, and `delegate` an attenuated CapabilitySet (amplification refused, Law L5).

## Features

- `frontend-python` / `frontend-rust` / `frontend-typescript` — forward to `hologram-compiler` to enable the corresponding source-language frontends for `compile`.

Part of the [hologram](../../README.md) workspace.
