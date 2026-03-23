//! Consumer-registered custom op registry.
//!
//! Allows downstream crates (e.g. `hologram-ai`) to extend the op set
//! without modifying `hologram` source.

use std::collections::HashMap;
use std::sync::Arc;

use hologram_graph::constant::ConstantStore;
use hologram_graph::graph::CustomOpId;

use crate::error::{ExecError, ExecResult};

/// A handler function for a custom op.
///
/// Receives the input byte buffers and the constant store; returns the output bytes.
pub type CustomHandler = Arc<dyn Fn(&[&[u8]], &ConstantStore) -> ExecResult<Vec<u8>> + Send + Sync>;

/// Registry mapping `CustomOpId`s to their handler functions.
///
/// Enables custom op dispatch for graphs containing `GraphOp::Custom` nodes.
#[derive(Default)]
pub struct CustomOpRegistry {
    handlers: HashMap<u32, CustomHandler>,
    arities: HashMap<u32, u8>,
}

impl CustomOpRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            arities: HashMap::new(),
        }
    }

    /// Register a handler for the given op id and arity.
    pub fn register(&mut self, id: CustomOpId, arity: u8, handler: CustomHandler) {
        self.handlers.insert(id.raw(), handler);
        self.arities.insert(id.raw(), arity);
    }

    /// Dispatch a custom op by id.
    ///
    /// Returns `Err(ExecError::UnsupportedOp)` if the id is not registered.
    pub fn dispatch(
        &self,
        id: CustomOpId,
        inputs: &[&[u8]],
        constants: &ConstantStore,
    ) -> ExecResult<Vec<u8>> {
        let handler = self
            .handlers
            .get(&id.raw())
            .ok_or_else(|| ExecError::UnsupportedOp(format!("custom op {}", id.raw())))?;
        handler(inputs, constants)
    }

    /// Whether the registry has no registered handlers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Number of registered handlers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handlers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> CustomOpRegistry {
        let mut r = CustomOpRegistry::new();
        r.register(
            CustomOpId(1),
            1,
            Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
        );
        r
    }

    #[test]
    fn registry_default_empty() {
        let r = CustomOpRegistry::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn unregistered_op_errors() {
        let r = CustomOpRegistry::new();
        let result = r.dispatch(CustomOpId(99), &[&[1, 2, 3]], &ConstantStore::new());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ExecError::UnsupportedOp(_)));
    }

    #[test]
    fn unregistered_with_registry_errors() {
        let r = make_registry();
        // id=1 registered, id=2 not
        let result = r.dispatch(CustomOpId(2), &[&[1]], &ConstantStore::new());
        assert!(result.is_err());
    }

    #[test]
    fn custom_passthrough() {
        let r = make_registry();
        let input = vec![10u8, 20, 30];
        let result = r
            .dispatch(CustomOpId(1), &[&input], &ConstantStore::new())
            .unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn multi_op_registry() {
        let mut r = CustomOpRegistry::new();
        r.register(
            CustomOpId(1),
            1,
            Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
        );
        r.register(
            CustomOpId(2),
            1,
            Arc::new(|inputs, _| Ok(inputs[0].iter().map(|&b| b.wrapping_mul(2)).collect())),
        );
        r.register(
            CustomOpId(3),
            2,
            Arc::new(|inputs, _| {
                Ok(inputs[0]
                    .iter()
                    .zip(inputs[1].iter())
                    .map(|(&a, &b)| a.wrapping_add(b))
                    .collect())
            }),
        );
        assert_eq!(r.len(), 3);

        let a = vec![1u8, 2, 3];
        assert_eq!(
            r.dispatch(CustomOpId(1), &[&a], &ConstantStore::new())
                .unwrap(),
            a
        );
        assert_eq!(
            r.dispatch(CustomOpId(2), &[&a], &ConstantStore::new())
                .unwrap(),
            vec![2, 4, 6]
        );
        let b = vec![10u8, 20, 30];
        assert_eq!(
            r.dispatch(CustomOpId(3), &[&a, &b], &ConstantStore::new())
                .unwrap(),
            vec![11, 22, 33]
        );
    }
}
