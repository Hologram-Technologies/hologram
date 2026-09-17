#!/usr/bin/env bash
#
# CC-47 (TARGET) — The lifecycle (suspend → κ snapshot → resume) works on every core
#
# OPM process: SD2 Lifecycle ("Suspending yields Snapshot; Resuming requires
# Snapshot → Holospace"). Today CC-30/CC-31 (suspend/resume from a κ snapshot)
# are witnessed on the RISC-V machine only. This target brings the AArch64 and
# x86-64 cores to lifecycle parity: a running guest suspends to a content-
# addressed snapshot (CPU + RAM + κ-disk) and resumes byte-identically.
#
# Authority: the CC-30 snapshot model (the substrate's content-addressed store as
#   the snapshot medium) applied to the AArch64 / x86-64 cores.
# Witness: crates/holospaces/tests/cc47_lifecycle_parity.rs — boot, run, suspend,
#   drop, resume; assert the resumed guest continues deterministically.
#
# GREEN when: suspend/resume round-trips on the AArch64 (and x86-64) cores — the
#   resumed CPU/RAM/disk state is byte-identical and execution continues.
#
# Status: LIVE — the canonical suite is release-gating.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/vv/suites/cc47-lifecycle-arch-parity.sh"
