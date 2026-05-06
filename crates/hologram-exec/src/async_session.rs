//! Async wrapper (folded from `hologram-async`, spec II.1 "deleted").

use crate::buffer::{InputBuffer, OutputBuffer};
use crate::error::ExecError;
use crate::session::{InferenceSession, SessionBackend};

/// Async wrapper around `InferenceSession::execute`.
///
/// Hologram's executor is synchronous in steady state; this entry point
/// exists so async runtimes (`tokio`, `async-std`) can drive inference
/// without blocking. It awaits zero work today — the body runs to
/// completion synchronously — but returning a future keeps the call site
/// in async context.
pub async fn execute_async<B: SessionBackend>(
    session: &mut InferenceSession<B>,
    inputs: &[InputBuffer<'_>],
) -> Result<Vec<OutputBuffer>, ExecError> {
    session.execute(inputs)
}
