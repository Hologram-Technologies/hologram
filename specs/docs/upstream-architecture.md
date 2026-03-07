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
| ADR-0001 | Repository boundaries and cross-repo isolation | hologram owns graph semantics and execution; sandbox/runtime targets live in hologram-sandbox |
| ADR-0002 | Canonical AI IR (`AiGraph`) above raw hologram graph | hologram-ai consumers lower to `hologram::Graph`; hologram has no AI format knowledge |
| ADR-0003 | Format-specific logic contained in importers | hologram receives format-agnostic graphs; no ONNX/GGUF/GGML types cross the boundary |
| ADR-0004 | Quantization is first-class; dequantization explicit | hologram-exec implements `QuantScheme` dispatch; no silent upcasting |
| ADR-0005 | InferenceSession owns plan + KV-cache; hologram owns execution | hologram exposes `KvExecutor` API; session state management is consumer responsibility |
| ADR-0006 | MVP scope: GGUF + CPU + single pass | hologram must provide complete CPU execution path before GPU backends |

---

## Local Interpretation

Hologram interprets upstream constraints as follows:

- **Graph abstraction boundary**: All AI-specific concepts (attention, KV-cache, tokens) terminate at the hologram-ai boundary. Hologram graphs contain only `GraphOp` nodes (Lut, Prim, FusedView, MatMulLut, Custom) with no semantic knowledge of what they compute.

- **Execution contract**: `KvExecutor::execute()` accepts a `Graph`, `ExecutionSchedule`, and `GraphInputs`. The caller (hologram-ai, hologram-sandbox, or direct user) owns buffer lifecycle and input preparation.

- **Quantization dispatch**: `hologram-exec` implements dequantization for all `QuantScheme` variants. Consumers register quantized-GEMM handlers via `CustomOpRegistry` if they want fused quantized kernels.

- **Archive format ownership**: `.holo` is hologram's format. Consumers (hologram-ai, hologram-sandbox) produce and consume archives but do not extend the format.

---

## Constraints This Repo Must Respect

1. **No AI format awareness**: hologram must not import, parse, or reference ONNX, GGUF, GGML, or any AI model format. These are hologram-ai concerns.

2. **No sandbox/runtime target implementation**: Process isolation, WASM sandboxing, and microVM targets belong in hologram-sandbox. hologram provides the execution substrate.

3. **One-way architecture flow**: This repo receives architecture decisions via `holoarch pull`. Local changes to `specs/docs/` are overwritten. Proposed architecture changes must go through hologram-architecture.

4. **Cross-repo isolation**: Agents working in hologram must not modify hologram-architecture, hologram-ai, hologram-sandbox, or any sibling repository. If cross-repo changes are needed, write a spec in `specs/plans/`.

5. **Explicit quantization**: All dequantization appears as explicit `AiOp::Dequantize` nodes or registered custom ops. No silent precision upgrades.

6. **Session state externality**: hologram does not own KV-cache, present_len, or multi-turn context. These are passed in via `GraphInputs` on each execution call.

7. **Backend-neutral interfaces**: hologram-exec provides CPU execution. GPU backends (Metal, CUDA, WebGPU) are future extensions that must not leak into core abstractions.