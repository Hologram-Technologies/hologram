# Upstream Architecture Reference — hologram

## Source of Truth

Architecture decisions for the Hologram ecosystem are maintained in:

```
hologram-architecture/
specs/adrs/          — Architecture Decision Records
specs/projects/      — Per-project planning docs
specs/research/      — Research reports
```

The ADRs in that repository are **authoritative**. Any constraint documented
there takes precedence over local conventions in `hologram`.

---

## Relevant ADRs

| ADR | Decision | Impact on hologram |
|-----|----------|------------------------|
| ADR-0001 | Keep hologram sandbox-agnostic | Runtime isolation (process, WASM, microVM) lives in `hologram-sandbox`, not here |
| ADR-0002 | Introduce canonical AiGraph IR | `hologram-ai` lowers to hologram types; hologram remains AI-agnostic |
| ADR-0005 | InferenceSession owns plan + KV-cache | hologram's `KvExecutor` is invoked by session; hologram has no KV-cache concept |
| ADR-0007 | Execution layer maps to real hologram types | `hologram-ai` uses `Graph`, `ExecutionSchedule`, `KvExecutor`, `BufferArena` directly |

---

## Local Interpretation

Hologram applies upstream constraints by maintaining strict boundaries:

- **AI-agnostic**: No AI-specific types (attention heads, KV-cache, tokenizers) appear in hologram. All such concepts stay in `hologram-ai`.
- **Sandbox-agnostic**: No process isolation, WASM runtime, or microVM code. Hologram provides portable execution primitives; isolation is layered on top.
- **Public API stability**: The types exposed for `hologram-ai` integration (`Graph`, `ExecutionSchedule`, `KvExecutor`, `BufferArena`, `CustomOpRegistry`) are the stable contract.

---

## Constraints This Repo Must Respect

1. **No AI format knowledge**: Hologram must not import ONNX, GGUF, GGML types or depend on AI model structure.
2. **No runtime isolation**: Hologram must not spawn processes, instantiate WASM runtimes, or manage sandboxes.
3. **Expose clean execution interface**: `KvExecutor::execute_with_registry()` is the primary entry point for external systems.
4. **No quantization semantics**: Quantized GEMM kernels exist (`MatMulLut4`, `MatMulLut8`), but quantization descriptors and dequant logic belong in `hologram-ai`.
5. **Portable core**: `hologram-core` must remain `no_std` compatible to support embedded and WASM targets.