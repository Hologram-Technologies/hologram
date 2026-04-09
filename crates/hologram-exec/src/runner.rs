//! Cooperative cancellation for tape execution.
//!
//! [`CancellationToken`] provides a wasm-compatible, lock-free signal
//! for cooperative cancellation. Pass it to `execute_direct` and the
//! executor checks it at level boundaries — if cancelled, it returns
//! `ExecError::Cancelled` without completing remaining instructions.
//!
//! No tokio dependency: uses `Arc<AtomicBool>` directly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cooperative cancellation signal, usable across threads.
///
/// wasm-compatible: no tokio or OS threading dependency.
/// Clone is cheap (Arc bump).
#[derive(Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Create a new token (not yet cancelled).
    #[inline]
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Signal cancellation. All clones observe this immediately.
    #[inline]
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_starts_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_is_visible() {
        let token = CancellationToken::new();
        let clone = token.clone();
        token.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn token_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CancellationToken>();
    }

    #[test]
    fn default_is_not_cancelled() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn debug_shows_state() {
        let token = CancellationToken::new();
        let dbg = format!("{token:?}");
        assert!(dbg.contains("false"));
        token.cancel();
        let dbg = format!("{token:?}");
        assert!(dbg.contains("true"));
    }
}
