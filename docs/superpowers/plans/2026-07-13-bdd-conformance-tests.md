# BDD Conformance Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a spec-first Gherkin/`cucumber` BDD conformance layer for the refactor, cross-linked to `CONFORMANCE.md`, with a bijection honesty meta-gate.

**Architecture:** A new leaf-tier test crate `crates/hologram-conformance` hosts the `cucumber` runner plus a static meta-gate. Gherkin `.feature` suites live at repo root under `features/suites/sN_*`. `CONFORMANCE.md` gains seven refactor classes (LAW/SP/HF/NW/TL/MG/GV); each catalog row maps 1:1 to a scenario, enforced by the meta-gate.

**Tech Stack:** Rust 2021, `cucumber` 0.21, `tokio` 1 (workspace dep), `just`.

## Global Constraints

- Workspace edition `2021`, version `0.9.0` (workspace-inherited). Copy verbatim: `version.workspace = true`, `edition.workspace = true`.
- `hologram-conformance` goes in `[workspace] members` but **NOT** `default-members` and **NOT** `exclude` — it must never enter the core build/publish graph.
- Leaf-tier rule (D22): the crate may depend on anything; nothing may depend on it.
- Status legend is single-sourced in `CONFORMANCE.md`: `✅` enforced & passing · `🟡` partial · `⛔` gap. The `@status` tag vocabulary is exactly `pending` (⛔) · `partial` (🟡) · `enforced` (✅).
- No `unwrap()`/`expect()` on non-test production paths; test/harness code may use them (law 7 exempts `#[cfg(test)]`). Harness parsing code returns `Result` where it is library code in `src/`.
- New classes and their scope, verbatim from the design spec:
  - **LAW** — repo laws (SPINE-1..6, κ-only identity, capability attenuation, async/sync, one surface) — spec 00
  - **SP** — space contract + TCK battery; external-repo parity (D21) — spec 02
  - **HF** — `.holo` v3 container, attenuated nesting, per-layer certificates — spec 03
  - **NW** — Network κ-realization, KappaSync/DHT, public/restricted/private tiers — spec 04
  - **TL** — one binary, one public facade crate, FFI over Client — spec 05
  - **MG** — phased always-green migration gates (P0–P6) — spec 06
  - **GV** — governance R1–R4 boundary rules — spec 07

---

### Task 1: Scaffold `hologram-conformance` crate + cucumber smoke run

**Files:**
- Create: `crates/hologram-conformance/Cargo.toml`
- Create: `crates/hologram-conformance/src/lib.rs`
- Create: `crates/hologram-conformance/tests/bdd.rs`
- Create: `features/suites/s0_laws/_smoke.feature`
- Modify: `Cargo.toml:22` (add member after `"crates/hologram-bench",`)

**Interfaces:**
- Produces: `hologram_conformance::ConformanceWorld` (async cucumber World, `Default`).
- Produces: test binary `bdd` runnable via `cargo test -p hologram-conformance --test bdd`.

- [ ] **Step 1: Add the workspace member**

Modify `Cargo.toml` — insert one line inside `members` immediately after line 22 (`    "crates/hologram-bench",`):

```toml
    "crates/hologram-bench",
    "crates/hologram-conformance",
```

Do **not** add it to `default-members` or `exclude`.

- [ ] **Step 2: Create the crate manifest**

Create `crates/hologram-conformance/Cargo.toml`:

```toml
[package]
name = "hologram-conformance"
version.workspace = true
edition.workspace = true
publish = false
description = "BDD (cucumber) conformance runner + honesty meta-gate for the hologram refactor."

[lib]
doctest = false

[dependencies]

[dev-dependencies]
cucumber = "0.21"
tokio = { workspace = true }

[[test]]
name = "bdd"
harness = false

[[test]]
name = "meta_gate"
harness = true
```

- [ ] **Step 3: Create the World**

Create `crates/hologram-conformance/src/lib.rs`:

```rust
//! BDD conformance harness for the hologram refactor.
//!
//! - `cucumber` runner entrypoint: `tests/bdd.rs`
//! - static honesty meta-gate: `tests/meta_gate.rs`
//! - catalog parser: [`catalog`]; feature parser: [`feature`]; gate: [`report`]
pub mod catalog;
pub mod feature;
pub mod report;

/// Absolute path to the repo-root `features/suites` tree, resolved at compile time.
pub const SUITES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../features/suites");

/// Absolute path to the repo-root `CONFORMANCE.md`.
pub const CONFORMANCE_MD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../CONFORMANCE.md");

/// Per-scenario async context. The only async↔sync seam (law 4): step bodies that
/// touch tensor compute stay synchronous inside this async World.
#[derive(Debug, Default, cucumber::World)]
pub struct ConformanceWorld {
    /// Set by a `When` step; asserted by a `Then` step. Placeholder until real
    /// contract handles land per phase.
    pub last_outcome: Option<String>,
}
```

Create empty module files so `lib.rs` compiles (filled in later tasks):

`crates/hologram-conformance/src/catalog.rs`:
```rust
//! Parser for the `CONFORMANCE.md` normative ledger. Filled in Task 2.
```
`crates/hologram-conformance/src/feature.rs`:
```rust
//! Parser for Gherkin `.feature` tags/scenarios. Filled in Task 3.
```
`crates/hologram-conformance/src/report.rs`:
```rust
//! Bijection honesty meta-gate. Filled in Task 6.
```

- [ ] **Step 4: Create the cucumber entrypoint**

Create `crates/hologram-conformance/tests/bdd.rs`:

```rust
//! Cucumber runner. Discovers every `.feature` under `features/suites`.
//!
//! Pending scenarios (no matching steps) are reported as skipped and do NOT fail
//! the run. As each phase (P0–P6) implements a suite, add its step definitions and
//! enable `.fail_on_skipped()` for that suite's tag (see features/README.md).
use hologram_conformance::ConformanceWorld;

#[tokio::main]
async fn main() {
    ConformanceWorld::run(hologram_conformance::SUITES_DIR).await;
}
```

- [ ] **Step 5: Create a smoke feature**

Create `features/suites/s0_laws/_smoke.feature`:

```gherkin
@class:LAW @id:LAW-0 @spec:00-overview @phase:P0 @status:enforced
Feature: Conformance harness smoke
  Scenario: the harness discovers and runs feature files
    Given the conformance harness is wired
    Then it runs at least one scenario
```

- [ ] **Step 6: Add step definitions for the smoke scenario**

Append to `crates/hologram-conformance/tests/bdd.rs` (above `main`):

```rust
use cucumber::{given, then};

#[given("the conformance harness is wired")]
fn harness_wired(_w: &mut ConformanceWorld) {}

#[then("it runs at least one scenario")]
fn runs_one(_w: &mut ConformanceWorld) {}
```

- [ ] **Step 7: Run the suite — verify it passes**

Run: `cargo test -p hologram-conformance --test bdd`
Expected: cucumber summary shows `1 scenario (1 passed)`, `2 steps (2 passed)`, exit 0.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml crates/hologram-conformance features/suites/s0_laws/_smoke.feature
git commit -m "test(conformance): scaffold hologram-conformance cucumber harness"
```

---

### Task 2: `catalog.rs` — parse CONFORMANCE.md rows

**Files:**
- Modify: `crates/hologram-conformance/src/catalog.rs`
- Test: `crates/hologram-conformance/src/catalog.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Produces: `pub enum Status { Enforced, Partial, Gap }` with `Status::from_legend(&str) -> Option<Status>`.
- Produces: `pub struct CatalogRow { pub class: String, pub id: String, pub status: Status }`.
- Produces: `pub fn parse_catalog(md: &str) -> Vec<CatalogRow>` — extracts every table row whose first cell is a bold id like `**LAW-1**`, reading the trailing status glyph.

- [ ] **Step 1: Write the failing test**

Replace `crates/hologram-conformance/src/catalog.rs` with:

```rust
//! Parser for the `CONFORMANCE.md` normative ledger.
//!
//! A row is a markdown table line whose first cell is a bold row id
//! (`| **LAW-1** | … | ✅ |`). We extract the id, its class prefix, and the
//! trailing status glyph.

/// Enforcement status, single-sourced from the CONFORMANCE.md legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Enforced,
    Partial,
    Gap,
}

impl Status {
    /// Map a legend glyph to a status. Returns `None` for anything else.
    pub fn from_legend(glyph: &str) -> Option<Status> {
        match glyph.trim() {
            "✅" => Some(Status::Enforced),
            "🟡" => Some(Status::Partial),
            "⛔" => Some(Status::Gap),
            _ => None,
        }
    }
}

/// One catalog row: class prefix, full id, declared status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogRow {
    pub class: String,
    pub id: String,
    pub status: Status,
}

/// Parse every id-bearing table row from a CONFORMANCE.md body.
pub fn parse_catalog(md: &str) -> Vec<CatalogRow> {
    let mut rows = Vec::new();
    for line in md.lines() {
        let line = line.trim();
        if !line.starts_with("| **") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // cells[0] is empty (leading pipe); cells[1] is the id cell.
        let Some(id_cell) = cells.get(1) else { continue };
        let Some(id) = id_cell.strip_prefix("**").and_then(|s| s.strip_suffix("**")) else {
            continue;
        };
        // id looks like LAW-1 / KC-6 / GV-3 — split on the first '-'.
        let Some((class, _)) = id.split_once('-') else { continue };
        // The last non-empty cell holds the status glyph.
        let Some(glyph) = cells.iter().rev().find(|c| !c.is_empty()) else { continue };
        let Some(status) = Status::from_legend(glyph) else { continue };
        rows.push(CatalogRow {
            class: class.to_string(),
            id: id.to_string(),
            status,
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
## LAW — repo laws
| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **LAW-1** | canonical bytes or nothing | BDD scenario | s0_laws/spine.feature | ⛔ |
| **LAW-2** | κ-only identity | BDD scenario | s0_laws/identity.feature | 🟡 |
| **KC-1** | matmul conforms | test | conformance.rs | ✅ |
";

    #[test]
    fn parses_ids_classes_and_status() {
        let rows = parse_catalog(FIXTURE);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], CatalogRow { class: "LAW".into(), id: "LAW-1".into(), status: Status::Gap });
        assert_eq!(rows[1].status, Status::Partial);
        assert_eq!(rows[2], CatalogRow { class: "KC".into(), id: "KC-1".into(), status: Status::Enforced });
    }

    #[test]
    fn ignores_non_row_lines() {
        assert!(parse_catalog("just prose\n| header | only |\n").is_empty());
    }
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p hologram-conformance --lib catalog`
Expected: PASS (`parses_ids_classes_and_status`, `ignores_non_row_lines`). The implementation is included above; if it fails, fix `catalog.rs` — do not edit the tests.

- [ ] **Step 3: Commit**

```bash
git add crates/hologram-conformance/src/catalog.rs
git commit -m "test(conformance): parse CONFORMANCE.md catalog rows"
```

---

### Task 3: `feature.rs` — parse Gherkin tags & scenarios

**Files:**
- Modify: `crates/hologram-conformance/src/feature.rs`
- Test: `crates/hologram-conformance/src/feature.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Produces: `pub struct ScenarioRef { pub class: String, pub id: String, pub status_tag: String, pub file: String, pub name: String }`.
- Produces: `pub fn parse_features(root: &std::path::Path) -> std::io::Result<Vec<ScenarioRef>>` — walks `*.feature`, reads `@class:/@id:/@status:` tags on each `Feature:` block.
- Produces: `pub fn status_tag_to_glyph(tag: &str) -> Option<&'static str>` mapping `pending→⛔`, `partial→🟡`, `enforced→✅`.

- [ ] **Step 1: Write the failing test**

Replace `crates/hologram-conformance/src/feature.rs` with:

```rust
//! Parser for Gherkin `.feature` files: extracts the `@class:/@id:/@status:` tags
//! that bind a Feature to a CONFORMANCE.md row.
use std::path::{Path, PathBuf};

/// A scenario's binding to a catalog row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioRef {
    pub class: String,
    pub id: String,
    pub status_tag: String,
    pub file: String,
    pub name: String,
}

/// Map a `@status:` tag to the CONFORMANCE.md legend glyph.
pub fn status_tag_to_glyph(tag: &str) -> Option<&'static str> {
    match tag {
        "pending" => Some("⛔"),
        "partial" => Some("🟡"),
        "enforced" => Some("✅"),
        _ => None,
    }
}

/// Extract the value of `@key:value` from a whitespace-split tag line.
fn tag_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(key))
}

/// Parse one feature file's leading tag block + Feature line into a `ScenarioRef`.
fn parse_one(path: &Path, body: &str) -> Option<ScenarioRef> {
    let mut class = None;
    let mut id = None;
    let mut status = None;
    let mut name = None;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('@') {
            class = class.or_else(|| tag_value(line, "@class:").map(str::to_string));
            id = id.or_else(|| tag_value(line, "@id:").map(str::to_string));
            status = status.or_else(|| tag_value(line, "@status:").map(str::to_string));
        } else if let Some(rest) = line.strip_prefix("Feature:") {
            name = Some(rest.trim().to_string());
            break;
        }
    }
    Some(ScenarioRef {
        class: class?,
        id: id?,
        status_tag: status?,
        file: path.file_name()?.to_string_lossy().into_owned(),
        name: name?,
    })
}

/// Walk `root` recursively and parse every `*.feature` file.
pub fn parse_features(root: &Path) -> std::io::Result<Vec<ScenarioRef>> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "feature") {
                let body = std::fs::read_to_string(&path)?;
                if let Some(sref) = parse_one(&path, &body) {
                    out.push(sref);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tag_block() {
        let body = "@class:GV @id:GV-1 @spec:07-governance @phase:P5 @status:pending\n\
                    Feature: Traceability by κ\n  Scenario: x\n    Given y\n";
        let sref = parse_one(Path::new("/x/trace.feature"), body).unwrap();
        assert_eq!(sref.class, "GV");
        assert_eq!(sref.id, "GV-1");
        assert_eq!(sref.status_tag, "pending");
        assert_eq!(sref.file, "trace.feature");
        assert_eq!(sref.name, "Traceability by κ");
    }

    #[test]
    fn status_glyph_mapping() {
        assert_eq!(status_tag_to_glyph("enforced"), Some("✅"));
        assert_eq!(status_tag_to_glyph("pending"), Some("⛔"));
        assert_eq!(status_tag_to_glyph("bogus"), None);
    }
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p hologram-conformance --lib feature`
Expected: PASS (`parses_tag_block`, `status_glyph_mapping`).

- [ ] **Step 3: Commit**

```bash
git add crates/hologram-conformance/src/feature.rs
git commit -m "test(conformance): parse Gherkin feature tag bindings"
```

---

### Task 4: Extend CONFORMANCE.md with the seven refactor classes + seed rows

**Files:**
- Modify: `CONFORMANCE.md` (append after the last existing class section; add class rows to the `## Classes` table)

**Interfaces:**
- Produces: catalog rows the meta-gate (Task 6) and feature suites (Task 5) bind to. Row ids MUST match Task 5 scenario tags exactly.

- [ ] **Step 1: Add the seven classes to the `## Classes` summary table**

In `CONFORMANCE.md`, append these rows to the table under `## Classes` (after `RP`):

```markdown
| **LAW** | Repo-wide laws (SPINE-1..6, κ-only identity, capability attenuation, async/sync, one surface) — refactor spec 00 | BDD scenarios (features/suites/s0_laws) |
| **SP** | Space contract trait set + laws + TCK battery; external-repo parity (D21) — spec 02 | BDD scenarios (s1_space_contract) |
| **HF** | `.holo` v3 container, attenuated nesting, per-layer certificates — spec 03 | BDD scenarios (s2_holo_format) |
| **NW** | Network κ-realization, KappaSync/DHT, public/restricted/private tiers — spec 04 | BDD scenarios (s3_networks) |
| **TL** | One binary, one public facade crate, FFI over Client — spec 05 | BDD scenarios (s4_tooling) |
| **MG** | Phased always-green migration gates (P0–P6) — spec 06 | BDD scenarios (s5_migration) |
| **GV** | Governance R1–R4 boundary rules — spec 07 | BDD scenarios (s6_governance) |
```

- [ ] **Step 2: Append the seed row sections**

Append to the end of `CONFORMANCE.md`. Every seed row is `⛔` (spec-only; not yet enforced). The `Witness` column points at the Task 5 feature file + scenario name.

```markdown
## LAW — repo-wide laws (refactor spec 00; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **LAW-1** | SPINE-1: a realization with no canonical bytes is unrepresentable; identity is verified by re-derivation, never trusted. | BDD scenario | `s0_laws/spine.feature::canonical bytes or nothing` | ⛔ |
| **LAW-2** | κ-only identity: no contract or stored form exposes a UUID / PeerId / Multiaddr / path / hostname as identity; transport ids never leak. | BDD scenario | `s0_laws/identity.feature::no second naming surface` | ⛔ |
| **LAW-5** | Capability attenuation only: a delegated capability is always a subset of the grantor's; amplification is unrepresentable. | BDD scenario | `s0_laws/attenuation.feature::delegation cannot amplify` | ⛔ |
| **LAW-6** | One programmatic surface: CLI / FFI / SDK are thin shells over the `Client` facade; behavior lives in exactly one place. | BDD scenario | `s0_laws/one_surface.feature::entry points are thin shells` | ⛔ |

## SP — space contract + TCK (refactor spec 02; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **SP-1** | Every space implements the identical contract surface; passing `hologram-tck` is the definition of conformance. | BDD scenario | `s1_space_contract/tck.feature::passing the TCK is conformance` | ⛔ |
| **SP-2** | An external-repo space passes the TCK as a dev-dependency and is accepted by `Client` with no facade change (D21). | BDD scenario | `s1_space_contract/external_parity.feature::external space is first-class` | ⛔ |

## HF — .holo v3 format (refactor spec 03; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **HF-1** | `.holo` v3 is the one application container; a tensor-only archive is the degenerate single-layer case. | BDD scenario | `s2_holo_format/container.feature::single format covers tensor-only` | ⛔ |
| **HF-2** | App nesting is capability-attenuated: a child's κ refs + delegated CapabilitySet are a subset of the parent's. | BDD scenario | `s2_holo_format/nesting.feature::nested app cannot exceed parent` | ⛔ |
| **HF-3** | v3 per-layer certificates verify; inspection APIs never strip them. | BDD scenario | `s2_holo_format/certificates.feature::per-layer certificates verify` | ⛔ |

## NW — networks (refactor spec 04; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **NW-1** | A Network is a κ-realization embedding its membership + policy operand κs (SPINE-2/3); no side tables. | BDD scenario | `s3_networks/realization.feature::network embeds operand κs` | ⛔ |
| **NW-2** | Network tiers (public / restricted / private) gate capability at the protocol boundary, never in business logic. | BDD scenario | `s3_networks/tiers.feature::tiers gate at the boundary` | ⛔ |

## TL — tooling (refactor spec 05; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **TL-1** | Exactly one binary named `hologram` ships. | BDD scenario | `s4_tooling/one_binary.feature::exactly one binary` | ⛔ |
| **TL-2** | Exactly one public crate (`hologram`) is imported with features; users never import subcrates. | BDD scenario | `s4_tooling/one_facade.feature::one public crate` | ⛔ |

## MG — migration gates (refactor spec 06; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **MG-1** | Every phase boundary P0–P6 is always-green: the full holospaces V&V passes before the next phase starts. | BDD scenario | `s5_migration/always_green.feature::each phase boundary is green` | ⛔ |
| **MG-2** | P0 sync exit criteria (D23) are met before any refactor move: holospaces ports to hologram HEAD, V&V green, bridge tag cut. | BDD scenario | `s5_migration/p0_sync.feature::p0 exit criteria met` | ⛔ |

## GV — governance requirements (refactor spec 07; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **GV-1** | R1 traceability: every new realization embeds its operand κs so `references()` yields the full provenance closure — no side tables. | BDD scenario | `s6_governance/traceability.feature::references yields full provenance` | ⛔ |
| **GV-2** | R2 auditability: lifecycle transitions emit through one seam that can be pointed at the κ-chain; no lifecycle path bypasses it. | BDD scenario | `s6_governance/auditability.feature::one audit seam, no bypass` | ⛔ |
| **GV-3** | R3 attestation: signing keys are bound to κ-addressed identities as published content; certificates are never a second identity surface. | BDD scenario | `s6_governance/attestation.feature::keys bind to κ-identity` | ⛔ |
| **GV-4** | R4 data governance: capability checks stay at the import/protocol boundary; resource accounting is per-capability, not global. | BDD scenario | `s6_governance/data_governance.feature::capability checks at the boundary` | ⛔ |
```

- [ ] **Step 3: Verify the catalog parser sees the new rows**

Run: `cargo test -p hologram-conformance --lib catalog`
Expected: still PASS (fixture-based; no regression).

- [ ] **Step 4: Commit**

```bash
git add CONFORMANCE.md
git commit -m "docs(conformance): add LAW/SP/HF/NW/TL/MG/GV classes + seed rows (all ⛔)"
```

---

### Task 5: Author the feature suites (all scenarios `@status:pending`)

**Files:**
- Create: `features/suites/s0_laws/{spine,identity,attenuation,one_surface}.feature`
- Create: `features/suites/s1_space_contract/{tck,external_parity}.feature`
- Create: `features/suites/s2_holo_format/{container,nesting,certificates}.feature`
- Create: `features/suites/s3_networks/{realization,tiers}.feature`
- Create: `features/suites/s4_tooling/{one_binary,one_facade}.feature`
- Create: `features/suites/s5_migration/{always_green,p0_sync}.feature`
- Create: `features/suites/s6_governance/{traceability,auditability,attestation,data_governance}.feature`

**Interfaces:**
- Consumes: catalog row ids from Task 4 (tags MUST match exactly).
- Produces: one pending scenario per seed row; the meta-gate (Task 6) binds them.

Every file follows this exact shape (tags → Feature → one Scenario with Given/When/Then). No step definitions exist yet, so cucumber reports each as skipped. **Scenario names MUST equal the `::name` in the Task 4 `Witness` cells.**

- [ ] **Step 1: Write the s0_laws features**

`features/suites/s0_laws/spine.feature`:
```gherkin
@class:LAW @id:LAW-1 @spec:00-overview §SPINE-1 @phase:P1 @status:pending
Feature: SPINE-1 — canonical bytes or nothing
  Scenario: canonical bytes or nothing
    Given a value with no canonical byte form
    When I attempt to construct a realization from it
    Then construction is unrepresentable and identity is only ever verified by re-derivation
```

`features/suites/s0_laws/identity.feature`:
```gherkin
@class:LAW @id:LAW-2 @spec:00-overview §law-2 @phase:P1 @status:pending
Feature: κ-only identity
  Scenario: no second naming surface
    Given a contract type and a stored realization
    When I enumerate every identity-bearing field
    Then none is a UUID, PeerId, Multiaddr, path, or hostname, and no transport id leaks
```

`features/suites/s0_laws/attenuation.feature`:
```gherkin
@class:LAW @id:LAW-5 @spec:00-overview §law-5 @phase:P2 @status:pending
Feature: capability attenuation only
  Scenario: delegation cannot amplify
    Given a capability set held by a grantor
    When the grantor delegates to a child
    Then the child's capabilities are a subset and amplification is unrepresentable
```

`features/suites/s0_laws/one_surface.feature`:
```gherkin
@class:LAW @id:LAW-6 @spec:00-overview §law-6 @phase:P3 @status:pending
Feature: one programmatic surface
  Scenario: entry points are thin shells
    Given the CLI, FFI, and SDK entry points
    When I trace each to where behavior is defined
    Then every path resolves to the single Client facade
```

- [ ] **Step 2: Write the s1_space_contract features**

`features/suites/s1_space_contract/tck.feature`:
```gherkin
@class:SP @id:SP-1 @spec:02-space-contract @phase:P2 @status:pending
Feature: space contract conformance
  Scenario: passing the TCK is conformance
    Given a space implementing the hologram-space traits
    When it runs the hologram-tck battery
    Then passing the TCK is the definition of conformance
```

`features/suites/s1_space_contract/external_parity.feature`:
```gherkin
@class:SP @id:SP-2 @spec:02-space-contract §D21 @phase:P4 @status:pending
Feature: external-repo space parity
  Scenario: external space is first-class
    Given a space living in an external repository depending only on published crates
    When it runs the TCK as a dev-dependency
    Then Client accepts it with no facade change
```

- [ ] **Step 3: Write the s2_holo_format features**

`features/suites/s2_holo_format/container.feature`:
```gherkin
@class:HF @id:HF-1 @spec:03-holo-format @phase:P3 @status:pending
Feature: .holo v3 is the one container
  Scenario: single format covers tensor-only
    Given a tensor-only archive
    When I open it as a .holo v3 application
    Then it is the degenerate single-layer case of the one format
```

`features/suites/s2_holo_format/nesting.feature`:
```gherkin
@class:HF @id:HF-2 @spec:03-holo-format §D9 @phase:P3 @status:pending
Feature: capability-attenuated app nesting
  Scenario: nested app cannot exceed parent
    Given a parent app with a CapabilitySet
    When it nests a child by κ ref with a delegated CapabilitySet
    Then the child's refs and capabilities are a subset of the parent's
```

`features/suites/s2_holo_format/certificates.feature`:
```gherkin
@class:HF @id:HF-3 @spec:03-holo-format @phase:P3 @status:pending
Feature: per-layer certificates
  Scenario: per-layer certificates verify
    Given a .holo v3 with per-layer certificates
    When I inspect it through the Client surface
    Then every certificate verifies and none is stripped
```

- [ ] **Step 4: Write the s3_networks features**

`features/suites/s3_networks/realization.feature`:
```gherkin
@class:NW @id:NW-1 @spec:04-networks §D12 @phase:P4 @status:pending
Feature: Network is a κ-realization
  Scenario: network embeds operand κs
    Given a Network built from a membership set and a policy
    When I call references() on its realization
    Then it yields the membership and policy operand κs with no side tables
```

`features/suites/s3_networks/tiers.feature`:
```gherkin
@class:NW @id:NW-2 @spec:04-networks @phase:P4 @status:pending
Feature: network tiers gate capability
  Scenario: tiers gate at the boundary
    Given public, restricted, and private network tiers
    When a peer attempts store/fetch/announce
    Then the capability check happens at the protocol boundary, not in business logic
```

- [ ] **Step 5: Write the s4_tooling features**

`features/suites/s4_tooling/one_binary.feature`:
```gherkin
@class:TL @id:TL-1 @spec:05-tooling §D13 @phase:P5 @status:pending
Feature: exactly one binary
  Scenario: exactly one binary
    Given the built workspace
    When I list installed binaries
    Then exactly one is named hologram
```

`features/suites/s4_tooling/one_facade.feature`:
```gherkin
@class:TL @id:TL-2 @spec:05-tooling §D4 @phase:P5 @status:pending
Feature: one public facade crate
  Scenario: one public crate
    Given a downstream consumer
    When it depends on the published crates
    Then it imports only the hologram facade with features, never a subcrate
```

- [ ] **Step 6: Write the s5_migration features**

`features/suites/s5_migration/always_green.feature`:
```gherkin
@class:MG @id:MG-1 @spec:06-migration §D17 @phase:P1 @status:pending
Feature: always-green phase boundaries
  Scenario: each phase boundary is green
    Given the refactor phase sequence P0 through P6
    When a phase boundary is reached
    Then the full holospaces V&V passes before the next phase starts
```

`features/suites/s5_migration/p0_sync.feature`:
```gherkin
@class:MG @id:MG-2 @spec:06-migration §D23 @phase:P0 @status:pending
Feature: P0 sync exit criteria
  Scenario: p0 exit criteria met
    Given holospaces pinned to its own repo
    When P0 completes
    Then holospaces ports to hologram HEAD, V&V is green, and the bridge tag is cut
```

- [ ] **Step 7: Write the s6_governance features**

`features/suites/s6_governance/traceability.feature`:
```gherkin
@class:GV @id:GV-1 @spec:07-governance §R1 @phase:P5 @status:pending
Feature: R1 traceability by κ
  Scenario: references yields full provenance
    Given a new realization built from known operand κs
    When I call references() on it
    Then the returned set equals the full provenance closure with no side tables
```

`features/suites/s6_governance/auditability.feature`:
```gherkin
@class:GV @id:GV-2 @spec:07-governance §R2 @phase:P5 @status:pending
Feature: R2 auditability
  Scenario: one audit seam, no bypass
    Given lifecycle transitions spawn, suspend, resume, terminate
    When each transition occurs
    Then it emits through one seam that can be pointed at the κ-chain and no path bypasses it
```

`features/suites/s6_governance/attestation.feature`:
```gherkin
@class:GV @id:GV-3 @spec:07-governance §R3 @phase:P5 @status:pending
Feature: R3 attestation
  Scenario: keys bind to κ-identity
    Given a space signing a session attestation
    When the signing key is published
    Then it is bound to a κ-addressed identity as content, never a second identity surface
```

`features/suites/s6_governance/data_governance.feature`:
```gherkin
@class:GV @id:GV-4 @spec:07-governance §R4 @phase:P5 @status:pending
Feature: R4 data governance
  Scenario: capability checks at the boundary
    Given a network capability policy with quotas
    When a peer stores, fetches, or announces content
    Then the capability check is at the import/protocol boundary and accounting is per-capability
```

- [ ] **Step 8: Run the full BDD suite — verify pending scenarios skip, run passes**

Run: `cargo test -p hologram-conformance --test bdd`
Expected: cucumber summary shows the smoke scenario passed and ~21 scenarios skipped (undefined steps); process exit 0.

- [ ] **Step 9: Commit**

```bash
git add features/suites
git commit -m "test(conformance): author LAW/SP/HF/NW/TL/MG/GV pending scenarios"
```

---

### Task 6: `report.rs` + `meta_gate.rs` — bijection honesty gate

**Files:**
- Modify: `crates/hologram-conformance/src/report.rs`
- Create: `crates/hologram-conformance/tests/meta_gate.rs`
- Test: `crates/hologram-conformance/src/report.rs` (inline `#[cfg(test)]` with drift fixtures)

**Interfaces:**
- Consumes: `catalog::{CatalogRow, Status, parse_catalog}`, `feature::{ScenarioRef, parse_features, status_tag_to_glyph}`.
- Produces: `pub const BDD_CLASSES: &[&str]` = the seven refactor classes.
- Produces: `pub fn check_bijection(rows: &[CatalogRow], scenarios: &[ScenarioRef]) -> Result<(), Vec<String>>` — returns the list of violations (empty Ok = clean).

- [ ] **Step 1: Write the failing test (drift fixtures)**

Replace `crates/hologram-conformance/src/report.rs` with:

```rust
//! The honesty meta-gate: a *static* check that the CONFORMANCE.md catalog and the
//! Gherkin scenarios are in bijection for the seven BDD classes, and that each row's
//! declared status glyph agrees with its scenario's `@status` tag.
use crate::catalog::CatalogRow;
use crate::feature::{status_tag_to_glyph, ScenarioRef};

/// The refactor classes whose rows are witnessed by BDD scenarios.
pub const BDD_CLASSES: &[&str] = &["LAW", "SP", "HF", "NW", "TL", "MG", "GV"];

fn is_bdd(class: &str) -> bool {
    BDD_CLASSES.contains(&class)
}

/// Check row↔scenario bijection + status agreement for the BDD classes only.
/// Returns `Err(violations)` when anything is off; `Ok(())` when clean.
pub fn check_bijection(
    rows: &[CatalogRow],
    scenarios: &[ScenarioRef],
) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();

    // Every BDD-class catalog row has exactly one scenario, with matching status.
    for row in rows.iter().filter(|r| is_bdd(&r.class)) {
        let matches: Vec<&ScenarioRef> = scenarios.iter().filter(|s| s.id == row.id).collect();
        match matches.as_slice() {
            [] => violations.push(format!("catalog row {} has no scenario", row.id)),
            [s] => {
                let want = status_tag_to_glyph(&s.status_tag);
                let have = crate::catalog::Status::from_legend(match row.status {
                    crate::catalog::Status::Enforced => "✅",
                    crate::catalog::Status::Partial => "🟡",
                    crate::catalog::Status::Gap => "⛔",
                });
                let scenario_status = want.and_then(crate::catalog::Status::from_legend);
                if scenario_status != have {
                    violations.push(format!(
                        "row {} status disagrees with scenario @status:{}",
                        row.id, s.status_tag
                    ));
                }
            }
            many => violations.push(format!(
                "catalog row {} has {} scenarios (want 1)",
                row.id,
                many.len()
            )),
        }
    }

    // Every BDD-class scenario names a catalog row that exists.
    for s in scenarios.iter().filter(|s| is_bdd(&s.class)) {
        if !rows.iter().any(|r| r.id == s.id) {
            violations.push(format!(
                "scenario {} ({}) names nonexistent catalog row {}",
                s.name, s.file, s.id
            ));
        }
        if status_tag_to_glyph(&s.status_tag).is_none() {
            violations.push(format!(
                "scenario {} has invalid @status:{}",
                s.name, s.status_tag
            ));
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Status;

    fn row(id: &str, status: Status) -> CatalogRow {
        CatalogRow { class: id.split_once('-').unwrap().0.to_string(), id: id.into(), status }
    }
    fn scn(id: &str, status_tag: &str) -> ScenarioRef {
        ScenarioRef {
            class: id.split_once('-').unwrap().0.to_string(),
            id: id.into(),
            status_tag: status_tag.into(),
            file: format!("{id}.feature"),
            name: id.into(),
        }
    }

    #[test]
    fn clean_when_in_bijection() {
        let rows = vec![row("GV-1", Status::Gap)];
        let scns = vec![scn("GV-1", "pending")];
        assert!(check_bijection(&rows, &scns).is_ok());
    }

    #[test]
    fn flags_row_without_scenario() {
        let rows = vec![row("GV-1", Status::Gap)];
        let err = check_bijection(&rows, &[]).unwrap_err();
        assert!(err.iter().any(|v| v.contains("has no scenario")));
    }

    #[test]
    fn flags_status_disagreement() {
        let rows = vec![row("GV-1", Status::Enforced)]; // ✅
        let scns = vec![scn("GV-1", "pending")]; // ⛔
        let err = check_bijection(&rows, &scns).unwrap_err();
        assert!(err.iter().any(|v| v.contains("status disagrees")));
    }

    #[test]
    fn ignores_non_bdd_classes() {
        let rows = vec![row("KC-1", Status::Enforced)]; // not a BDD class
        assert!(check_bijection(&rows, &[]).is_ok());
    }
}
```

- [ ] **Step 2: Run the unit tests to verify they pass**

Run: `cargo test -p hologram-conformance --lib report`
Expected: PASS (4 tests).

- [ ] **Step 3: Create the real meta-gate integration test**

Create `crates/hologram-conformance/tests/meta_gate.rs`:

```rust
//! Runs the honesty meta-gate against the real CONFORMANCE.md + features tree.
//! Fails the build if the catalog and scenarios drift out of bijection.
use hologram_conformance::{catalog, feature, report, CONFORMANCE_MD, SUITES_DIR};
use std::path::Path;

#[test]
fn catalog_and_scenarios_are_in_bijection() {
    let md = std::fs::read_to_string(CONFORMANCE_MD).expect("read CONFORMANCE.md");
    let rows = catalog::parse_catalog(&md);
    let scenarios = feature::parse_features(Path::new(SUITES_DIR)).expect("parse features");

    if let Err(violations) = report::check_bijection(&rows, &scenarios) {
        panic!(
            "conformance honesty meta-gate failed:\n  - {}",
            violations.join("\n  - ")
        );
    }
}
```

- [ ] **Step 4: Run the meta-gate against real files**

Run: `cargo test -p hologram-conformance --test meta_gate`
Expected: PASS. If it fails, the panic lists exactly which row/scenario is out of bijection — fix the mismatch in `CONFORMANCE.md` (Task 4) or the `.feature` tag (Task 5), never the gate. Note: the smoke row `LAW-0` is `@status:enforced` but has no catalog row; either add a `LAW-0` row (`✅`) to CONFORMANCE.md or retag the smoke feature to a non-BDD class. Add the `LAW-0` row to keep it self-consistent:

```markdown
| **LAW-0** | Harness smoke: the conformance runner discovers and executes feature files. | BDD scenario | `s0_laws/_smoke.feature::the harness discovers and runs feature files` | ✅ |
```

Re-run the meta-gate; expected PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/hologram-conformance/src/report.rs crates/hologram-conformance/tests/meta_gate.rs CONFORMANCE.md
git commit -m "test(conformance): bijection honesty meta-gate (catalog ↔ scenarios)"
```

---

### Task 7: Wire `just` targets, README, fold into `vv`

**Files:**
- Modify: `Justfile` (add `bdd` and `conformance-report` recipes; add `bdd` to `vv`)
- Create: `features/README.md`

**Interfaces:**
- Produces: `just bdd`, `just conformance-report`; `just vv` now includes `bdd`.

- [ ] **Step 1: Add the `bdd` recipe**

In `Justfile`, add after the `conformance:` recipe block (ends at the `hologram-exec --test conformance` line):

```make
# BDD conformance suite (refactor classes LAW/SP/HF/NW/TL/MG/GV). Runs the cucumber
# runner + the honesty meta-gate (catalog ↔ scenario bijection). See features/README.md.
bdd:
    cargo test -p hologram-conformance --test bdd
    cargo test -p hologram-conformance --test meta_gate

# Regenerate + verify the BDD status column against actual scenario tags. Fails on drift.
conformance-report:
    cargo test -p hologram-conformance --test meta_gate
```

- [ ] **Step 2: Fold `bdd` into `vv`**

Modify the `vv:` line in `Justfile`:

```make
vv: fmt-check clippy test conformance bdd parallel perf wasm embedded
    @echo "V&V complete — see CONFORMANCE.md for the invariant catalog."
```

- [ ] **Step 3: Write features/README.md**

Create `features/README.md`:

```markdown
# Conformance BDD suites

Gherkin `.feature` suites for the hologram **refactor** (`specs/refactor/00`–`07`),
run by the `cucumber` crate in `crates/hologram-conformance`. Modeled on
`afflom/UOR-Atlas-UTQC`'s `features/suites`, informed by and cross-linked to the
root `CONFORMANCE.md` normative ledger.

## Layout

- `suites/s0_laws` … `suites/s6_governance` — one suite per refactor spec area.
- Each scenario is tagged `@class:<C> @id:<C-N> @spec:<doc> @phase:<Pn> @status:<s>`.
- `@class`/`@id` bind the scenario to a `CONFORMANCE.md` row (classes LAW/SP/HF/NW/TL/MG/GV).

## Status vocabulary (cross-walked to the CONFORMANCE.md legend)

| `@status` | scenario | catalog |
|---|---|---|
| `pending` | steps skip (undefined) | ⛔ gap |
| `partial` | some steps assert | 🟡 partial |
| `enforced` | all steps assert & pass | ✅ enforced |

## Running

- `just bdd` — run the suite + the honesty meta-gate.
- `just conformance-report` — verify the catalog ↔ scenario bijection (fails on drift).

## Honesty rule

The meta-gate (`crates/hologram-conformance/tests/meta_gate.rs`) enforces that every
BDD-class catalog row has exactly one scenario, every scenario names a real row, and the
row's status glyph matches the scenario's `@status`. **No requirement is "done" until its
scenario is green and CI-gated.**

## Phased rollout

Scenarios land `pending` and turn `enforced` as the phase in their `@phase:` tag
implements the requirement. When a phase implements a suite, add step definitions in
`crates/hologram-conformance/tests/bdd.rs`, flip the `@status` tag to `enforced`, update
the matching `CONFORMANCE.md` row to `✅`, and enable `.fail_on_skipped()` for that
suite's tag so an enforced-but-unimplemented scenario fails the build.
```

- [ ] **Step 4: Verify the wired targets**

Run: `just bdd`
Expected: both `bdd` and `meta_gate` test binaries run and pass (smoke green, refactor scenarios skipped, meta-gate green).

Run: `just conformance-report`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Justfile features/README.md
git commit -m "build(conformance): just bdd + conformance-report, fold bdd into vv"
```

---

## Self-Review

**Spec coverage:**
- Section 1 (architecture/layout) → Tasks 1, 7 (crate, features tree, README).
- Section 2 (catalog linkage, 7 classes, bijection) → Tasks 2, 4, 6.
- Section 3 (tags + status axis) → Tasks 3, 5 (tags), 6 (status agreement).
- Section 4 (runner, just targets, phased rollout) → Tasks 1, 7.
- Components/isolation view → catalog.rs (T2), feature.rs (T3), report.rs+meta_gate.rs (T6), steps/World (T1), features (T5).
- Error handling → T6 (bijection violations listed), T3 (parse returns `io::Result`).
- Harness self-tests → T2, T3, T6 (drift fixtures).

**Placeholder scan:** none — every step ships complete code or exact file content.

**Type consistency:** `CatalogRow{class,id,status}`, `Status{Enforced,Partial,Gap}`, `ScenarioRef{class,id,status_tag,file,name}`, `parse_catalog`, `parse_features`, `status_tag_to_glyph`, `check_bijection`, `BDD_CLASSES`, `SUITES_DIR`, `CONFORMANCE_MD`, `ConformanceWorld` — used identically across Tasks 1–7. Row ids in Task 4 match scenario `@id` tags in Task 5 and `Witness` `::name` values match Task 5 scenario names.
