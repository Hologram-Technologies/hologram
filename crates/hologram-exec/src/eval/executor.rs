//! Graph execution types: inputs, outputs, and runtime context.

use std::collections::HashMap;

/// Runtime context passed to dispatch during execution.
///
/// Carries execution-time state that cannot be baked into the compiled graph
/// (e.g., the current token position for RoPE during KV cache decode).
/// Non-KV execution passes `None` — zero overhead.
pub struct ExecutionContext {
    /// Position offset for positional encodings (RoPE).
    /// Set from `KvCacheState::write_pos()` at the start of each call.
    /// 0 during prefill, N during decode (N = tokens already cached).
    pub position_offset: u32,
}

/// Configuration for batch-aware scheduling.
///
/// Supports continuous batching: multiple sequences share a KV cache prefix
/// and diverge at `shared_prefix_len`. The scheduler can overlap prefill of
/// new sequences with decode of in-flight sequences, amortizing attention
/// cost over the shared prefix region.
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Number of sequences in the batch.
    pub batch_size: usize,
    /// Number of tokens in the shared KV cache prefix.
    /// Tokens `0..shared_prefix_len` are computed once and reused.
    pub shared_prefix_len: usize,
}

/// Named graph inputs: maps input index to byte data and optional shape.
#[derive(Debug, Clone, Default)]
pub struct GraphInputs {
    inputs: HashMap<u32, Vec<u8>>,
    shapes: HashMap<u32, Vec<usize>>,
}

impl GraphInputs {
    /// Create empty inputs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Set data for graph input at `index`.
    pub fn set(&mut self, index: u32, data: Vec<u8>) {
        self.inputs.insert(index, data);
    }

    /// Set data with an explicit N-D shape for graph input at `index`.
    pub fn set_with_shape(&mut self, index: u32, data: Vec<u8>, shape: Vec<usize>) {
        self.inputs.insert(index, data);
        self.shapes.insert(index, shape);
    }

    /// Get data for graph input at `index`.
    pub fn get(&self, index: u32) -> Option<&[u8]> {
        self.inputs.get(&index).map(|v| v.as_slice())
    }

    /// Get the shape for graph input at `index`, if set.
    pub fn shape(&self, index: u32) -> Option<&[usize]> {
        self.shapes.get(&index).map(|v| v.as_slice())
    }

    /// Create from a list of (index, data) pairs.
    #[must_use]
    pub fn from_pairs(pairs: Vec<(u32, Vec<u8>)>) -> Self {
        Self {
            inputs: pairs.into_iter().collect(),
            shapes: HashMap::new(),
        }
    }
}

/// Named graph outputs: list of (name, data) pairs.
#[derive(Debug, Clone, Default)]
pub struct GraphOutputs {
    outputs: Vec<(String, Vec<u8>)>,
}

impl GraphOutputs {
    /// Number of outputs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Whether there are no outputs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Get output by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(&str, &[u8])> {
        self.outputs
            .get(index)
            .map(|(name, data)| (name.as_str(), data.as_slice()))
    }

    /// Get output by name.
    #[must_use]
    pub fn by_name(&self, name: &str) -> Option<&[u8]> {
        self.outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// Consume into inner vec.
    #[must_use]
    pub fn into_inner(self) -> Vec<(String, Vec<u8>)> {
        self.outputs
    }

    /// Create from named output pairs.
    #[must_use]
    pub fn from_named(outputs: Vec<(String, Vec<u8>)>) -> Self {
        Self { outputs }
    }
}
