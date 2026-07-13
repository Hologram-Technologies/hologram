//! Cucumber runner. Discovers every `.feature` under `features/suites`.
//!
//! Pending scenarios (no matching steps) are reported as skipped and do NOT fail
//! the run. As each phase (P0–P6) implements a suite, add its step definitions and
//! enable `.fail_on_skipped()` for that suite's tag (see features/README.md).
use hologram_conformance::ConformanceWorld;

use cucumber::{given, then, World};

#[given("the conformance harness is wired")]
fn harness_wired(_w: &mut ConformanceWorld) {}

#[then("it runs at least one scenario")]
fn runs_one(_w: &mut ConformanceWorld) {}

#[tokio::main]
async fn main() {
    ConformanceWorld::run(hologram_conformance::SUITES_DIR).await;
}
