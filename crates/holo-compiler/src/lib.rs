//! Compilation pipeline: Graph → optimized .holo archive.
//!
//! Three-stage pipeline:
//! 1. **Parse**: validate graph structure
//! 2. **Fuse**: optimization pass (constant folding, view fusion, CSE)
//! 3. **Plan & Emit**: schedule, liveness, workspace planning, .holo emission

pub mod compiler;
pub mod error;
pub mod liveness;
pub mod workspace;

pub use compiler::{compile, CompilationOutput, CompilationStats, CompilerBuilder};
pub use error::{CompileError, CompileResult};
pub use liveness::LivenessInterval;
pub use workspace::WorkspaceLayout;
