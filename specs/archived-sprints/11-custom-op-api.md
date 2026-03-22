# Sprint 11: Custom Op Extension API

**Completed**: 2026-03-06
**Tests**: 700 total workspace tests, zero clippy warnings

## Goal

Allow `hologram-ai` to register custom ops at compile time without modifying `hologram` source. Avoids global mutable state by threading a `CustomOpRegistry` through the execution API.

## Deliverables

- [x] `CustomOpId(pub u32)` newtype with rkyv derives in `hologram-graph/src/graph/mod.rs`
- [x] `GraphOp::Custom { id: CustomOpId, arity: u8 }` variant; updated `arity`, `is_pure`, `to_view`
- [x] `CustomOpId` re-exported from `hologram-graph/src/lib.rs`
- [x] `GraphBuilder::custom_op(id, arity, inputs)` builder method in `hologram-graph/src/builder/mod.rs`
- [x] `CustomHandler = Arc<dyn Fn(&[&[u8]], &ConstantStore) -> ExecResult<Vec<u8>> + Send + Sync>`
- [x] `CustomOpRegistry` in `hologram-exec/src/kv/registry.rs`: `register`, `dispatch`, `len`, `is_empty`, `Default`
- [x] `register_op!(registry, id = N, arity = A, handler = ...)` macro in `hologram-exec/src/lib.rs`
- [x] `KvStore::dispatch` and `dispatch_with_constants` accept `Option<&CustomOpRegistry>`
- [x] Registry threaded through private `execute_core` → `dispatch_level` → `KvStore::dispatch_with_constants`
- [x] `KvExecutor::execute_with_registry` new public method (no progress callback)
- [x] `execute_bytes_with_ops(data, inputs, registry)` new public API in `hologram-exec/src/mmap/mod.rs`
- [x] `hologram-exec/src/lib.rs`: exports `CustomOpId`, `CustomHandler`, `CustomOpRegistry`, `execute_bytes_with_ops`, `register_op!`
- [x] Sprint 10 archived to `specs/sprints/10-cli-completeness.md`
- [x] 11 integration tests in `tests/custom_ops.rs`; 4 unit tests in `registry.rs`; zero clippy warnings; `just ci` green — **700 total workspace tests**

## Implementation Notes

- `CustomOpRegistry` uses `HashMap<u32, CustomHandler>` keyed on `CustomOpId::raw()` — no global state.
- Unregistered op with no registry → `ExecError::UnsupportedOp("custom op N")`.
- `GraphOp::Custom` serializes cleanly via rkyv (id + arity only); the handler is NOT serialized — consumer re-registers at startup.
- `execute_with_progress` signature unchanged; new `execute_core` private helper adds the registry parameter.
- All existing callers of `KvStore::dispatch` / `dispatch_with_constants` pass `None` — zero breaking changes to existing tests.
- ONNX has ~170 standard operators; Hologram's LUT model only fits elementwise ops. `register_op!` is the right abstraction at 20–60 op scale. Full descriptor-based codegen deferred to Sprint 12+ (threshold: 80+ ops).
