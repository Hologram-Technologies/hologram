# Plan: Enum Dispatch + LUT-GEMM Tape Wiring

## Context

Two changes combined into one pass since both touch the `BoxedKernel` type:

1. **Enum dispatch** — Replace `Box<dyn Fn>` closures with a `TapeKernel` enum + match dispatch. Eliminates vtable indirection (~3ns/call), enables inlining of small kernels, and removes per-kernel heap allocation. Most impactful for hologram's byte-domain LUT compute where kernels are fast.

2. **LUT-GEMM wiring** — Replace the 4 error-returning LUT-GEMM closures with real kernel dispatch that resolves quantized weights via `WeightCache`. Enables the optimized tape path for quantized inference (used by hologram-ai).

## Design: `TapeKernel` Enum

```rust
// tape.rs
pub enum TapeKernel {
    /// Float op dispatched via dispatch_float_into (covers elementwise, matmul, softmax, etc.)
    Float(FloatOp),
    /// Fused chain of unary float ops.
    FusedFloatChain(Vec<FloatOp>),
    /// Graph output passthrough.
    Output,
    /// Byte-domain LUT (256-byte table).
    LutView(ElementWiseView),
    /// Byte-domain unary prim via LUT.
    PrimUnary(ElementWiseView),
    /// Byte-domain binary prim.
    PrimBinary(PrimOp),
    /// 4-bit quantized LUT-GEMM matmul.
    MatMulLut4(ConstantId),
    /// 8-bit quantized LUT-GEMM matmul.
    MatMulLut8(ConstantId),
}
```

## Design: `TapeContext`

```rust
// tape.rs
pub struct TapeContext<'a> {
    pub ctx: Option<ExecutionContext>,
    pub constants: &'a ConstantStore,
    pub weights: &'a [u8],
    pub weight_cache: RefCell<WeightCache>,
}

impl TapeContext<'_> {
    /// Create a minimal context (no weights/constants — for float-only tapes).
    pub fn empty() -> TapeContext<'static> { ... }
}
```

## Design: Dispatch Function

```rust
// tape.rs
fn dispatch_kernel(
    kernel: &TapeKernel,
    inputs: &[&[u8]],
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    match kernel {
        TapeKernel::Float(op) => {
            float_dispatch::dispatch_float_into(op, inputs, tape_ctx.ctx.as_ref(), out_buf)
        }
        TapeKernel::FusedFloatChain(chain) => {
            float_dispatch::dispatch_fused_chain_into(chain, inputs, out_buf)
        }
        TapeKernel::Output => {
            if let Some(b) = inputs.first() { out_buf.extend_from_slice(b); }
            Ok(())
        }
        TapeKernel::LutView(view) | TapeKernel::PrimUnary(view) => {
            out_buf.extend_from_slice(&KvStore::apply_unary(view, inputs[0]));
            Ok(())
        }
        TapeKernel::PrimBinary(p) => {
            let r = KvStore::apply_binary(*p, inputs[0], inputs[1])?;
            out_buf.extend_from_slice(&r);
            Ok(())
        }
        TapeKernel::MatMulLut4(cid) => {
            let mut cache = tape_ctx.weight_cache.borrow_mut();
            let qw = cache.get_q4(*cid, tape_ctx.constants, tape_ctx.weights)?;
            let activations: &[f32] = bytemuck::cast_slice(inputs[0]);
            let m = activations.len() / qw.rows as usize;
            let mut output = vec![0.0f32; m * qw.cols as usize];
            lut_gemm_4bit(activations, qw, &mut output);
            out_buf.extend_from_slice(bytemuck::cast_slice(&output));
            Ok(())
        }
        TapeKernel::MatMulLut8(cid) => {
            // Same pattern with get_q8 + lut_gemm_8bit
        }
    }
}
```

## Updated Instruction Struct

```rust
pub struct TapeInstruction {
    pub kernel: TapeKernel,       // was: BoxedKernel (Box<dyn Fn>)
    pub output_idx: u32,
    pub input_indices: Vec<u32>,
    pub output_elem_size: u8,
    pub output_byte_hint: u32,
}
```

Rename from `BoxedInstruction` to `TapeInstruction` since it's no longer boxed.

## Updated Execute Loop

```rust
pub fn execute(
    &self,
    arena: &mut BufferArena<'_>,
    tape_ctx: &TapeContext<'_>,
) -> ExecResult<()> {
    let mut out_buf = Vec::with_capacity(4096);
    for (i, instr) in self.instructions.iter().enumerate() {
        // prefetch...
        {
            let input_refs = ...collect()?;
            out_buf.clear();
            if instr.output_byte_hint > 0 { out_buf.reserve(instr.output_byte_hint as usize); }
            dispatch_kernel(&instr.kernel, &input_refs, tape_ctx, &mut out_buf)?;
        }
        arena.swap_insert_with_elem_size(out_id, &mut out_buf, instr.output_elem_size as usize);
    }
    Ok(())
}
```

## Files to Modify

| File | Change |
|------|--------|
| `crates/hologram-exec/src/tape.rs` | Replace `BoxedKernel`/`BoxedInstruction` with `TapeKernel`/`TapeInstruction`, add `TapeContext`, add `dispatch_kernel`, update execute loops, remove `resolve_boxed_kernel`, update tests |
| `crates/hologram-exec/src/tape_builder.rs` | `resolve_kernel` returns `TapeKernel` enum variants (not closures), `resolve_float_kernel` returns `TapeKernel::Float(op)`, remove all `Box::new(move \|...\|)` closures |
| `crates/hologram-exec/src/mmap/mod.rs` | Build `TapeContext` in `execute_tape`/`build_tape_from_plan`, pass to `execute` |
| `specs/SPRINT.md` | Add Phase 8 tasks |
| `specs/plans/008-enum-dispatch-lut-gemm.md` | This plan |

## What Gets Deleted

- `BoxedKernel` type alias (replaced by `TapeKernel` enum)
- `BoxedInstruction` struct (replaced by `TapeInstruction`)
- `resolve_boxed_kernel()` function
- All 19 `Box::new(move |inputs, _ctx, out_buf| { ... })` closures in tape_builder
- `KernelFn` type alias (only used in `Instruction` which uses fn pointers — keep for now or also migrate)

## What Does NOT Change

- `dispatch_float_into` and all `_into` variants — called from `dispatch_kernel`
- LUT-GEMM kernels, WeightCache, Psumbook — used unchanged
- KvExecutor path — completely unaffected
- `BufferArena`, swap-insert — unchanged
- Archive format, graph format — unchanged

## Verification

- `cargo test --workspace` — all tests pass
- `cargo clippy --workspace -- -D warnings` — zero warnings
- `just bench` — compare tape executor benchmarks (expect small improvement from inlining)
