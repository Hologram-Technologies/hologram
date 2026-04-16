//! Metal compute backend (Apple GPU).
//!
//! Implements `MetalMemory` and `MetalBackend` for device-native execution
//! on Apple Silicon. All tensor data lives in Metal shared-memory buffers.
//! All computation dispatches as Metal compute shaders.

// TODO: MetalMemory + MetalBackend implementation (Plan 067 Phase 2).
