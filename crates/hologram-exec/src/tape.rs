//! Instruction tape executor for zero-match dispatch.
//!
//! The tape is a flat array of pre-resolved instructions compiled from
//! the graph's execution schedule. Each instruction stores a kernel function
//! pointer and pre-resolved input/output indices, eliminating the large
//! `match op { ... }` dispatch at runtime.
//!
//! The tape is built once per model load and executed per inference call.
//! This is Phase 0.7 of the Compile-Time-First Acceleration plan.

use smallvec::SmallVec;

use hologram_core::op::FloatOp;
use hologram_graph::graph::node::NodeId;

use crate::buffer::BufferArena;
use crate::error::ExecResult;
use crate::eval::executor::ExecutionContext;

/// Non-blocking prefetch of a cache line into L1 for reading.
///
/// Uses platform-specific intrinsics where available:
/// - x86_64: `_mm_prefetch(..., _MM_HINT_T0)` (L1 temporal)
/// - aarch64: `PRFM PLDL1KEEP` via inline asm
/// - Other: no-op (rely on hardware prefetcher)
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// A kernel function: takes input byte slices, optional context, writes output
/// into a pre-provided buffer. The buffer is cleared before each call but
/// retains its heap allocation across instructions.
pub type KernelFn = fn(&[&[u8]], Option<&ExecutionContext>, &mut Vec<u8>) -> ExecResult<()>;

/// A single instruction in the execution tape.
pub struct Instruction {
    /// The kernel to execute.
    pub kernel: KernelFn,
    /// Output node index (where to store the result in the arena).
    pub output_idx: u32,
    /// Input node indices (where to gather inputs from the arena).
    pub input_indices: Vec<u32>,
    /// Element size of the output (for arena metadata).
    pub output_elem_size: u8,
    /// Graph-specific tile size hint (Phase 12.3). 0 = no hint.
    /// Used by tiled kernels (LUT-GEMM, conv2d) to select tile dimensions.
    pub tile_hint: u16,
}

/// Pre-compiled execution tape.
///
/// Built once from a graph + schedule, then executed repeatedly per inference.
/// Each execution reuses the same tape with different arena contents.
pub struct Tape {
    /// Flat instruction array in execution order (level-by-level, sequential within level).
    pub instructions: Vec<Instruction>,
    /// Level boundaries: `level_offsets[i]..level_offsets[i+1]` is the range of
    /// instructions for level `i`. Used for parallel execution of levels.
    pub level_offsets: Vec<usize>,
}

impl Tape {
    /// Create an empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            level_offsets: vec![0],
        }
    }

    /// Create a tape with pre-allocated instruction capacity.
    #[must_use]
    pub fn with_capacity(n_instructions: usize, n_levels: usize) -> Self {
        let mut level_offsets = Vec::with_capacity(n_levels + 1);
        level_offsets.push(0);
        Self {
            instructions: Vec::with_capacity(n_instructions),
            level_offsets,
        }
    }

    /// Add an instruction and return its index.
    pub fn push(&mut self, instr: Instruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Mark the end of the current level.
    pub fn end_level(&mut self) {
        self.level_offsets.push(self.instructions.len());
    }

    /// Number of levels in the tape.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.level_offsets.len().saturating_sub(1)
    }

    /// Execute the tape sequentially against the given arena.
    ///
    /// A single reusable output buffer is swapped between the kernel and the
    /// arena via [`BufferArena::swap_insert_with_elem_size`]. After warmup,
    /// this eliminates per-instruction heap allocation entirely.
    pub fn execute(
        &self,
        arena: &mut BufferArena<'_>,
        ctx: Option<&ExecutionContext>,
    ) -> ExecResult<()> {
        let mut out_buf: Vec<u8> = Vec::with_capacity(4096);

        for (i, instr) in self.instructions.iter().enumerate() {
            // Prefetch next instruction's input data into cache.
            if i + 1 < self.instructions.len() {
                let next = &self.instructions[i + 1];
                for &idx in &next.input_indices {
                    let id = NodeId::new(idx, 0);
                    if let Ok(data) = arena.get(id) {
                        prefetch_read(data.as_ptr());
                    }
                }
            }

            // Gather input refs and execute kernel in a scoped block.
            // The immutable borrow of `arena` is released when the block ends.
            {
                let input_refs: SmallVec<[&[u8]; 4]> = instr
                    .input_indices
                    .iter()
                    .map(|&idx| arena.get(NodeId::new(idx, 0)))
                    .collect::<ExecResult<SmallVec<_>>>()?;
                out_buf.clear();
                (instr.kernel)(&input_refs, ctx, &mut out_buf)?;
            }

            // Swap output into arena — recycles the old buffer's allocation.
            let out_id = NodeId::new(instr.output_idx, 0);
            arena.swap_insert_with_elem_size(out_id, &mut out_buf, instr.output_elem_size as usize);
        }

        Ok(())
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

/// A boxed kernel: like `KernelFn` but can capture op parameters.
///
/// Used for ops that need baked-in parameters (e.g., Softmax with size,
/// MatMul with m/k/n). The Box<dyn Fn> has one indirection but eliminates
/// the `match op` at execution time — the parameters are pre-resolved.
///
/// Writes output into a pre-provided `&mut Vec<u8>` instead of returning
/// a new allocation. The buffer is cleared before each call.
pub type BoxedKernel =
    Box<dyn Fn(&[&[u8]], Option<&ExecutionContext>, &mut Vec<u8>) -> ExecResult<()> + Send + Sync>;

/// Instruction variant that uses a boxed kernel (captures op parameters).
pub struct BoxedInstruction {
    pub kernel: BoxedKernel,
    pub output_idx: u32,
    pub input_indices: Vec<u32>,
    pub output_elem_size: u8,
    /// Pre-computed output byte size hint (0 = unknown/dynamic).
    ///
    /// When non-zero, the executor pre-reserves this many bytes in the output
    /// buffer before calling the kernel, reducing reallocation pressure.
    /// Computed from compiled node shapes + dtypes at tape build time.
    pub output_byte_hint: u32,
}

/// Resolve a FloatOp to a boxed kernel that captures its parameters.
///
/// This is the "compile" step: bakes op-specific parameters into the closure,
/// eliminating the match dispatch at execution time.
pub fn resolve_boxed_kernel(op: &FloatOp) -> BoxedKernel {
    use crate::float_dispatch;

    let op = *op;
    Box::new(move |inputs, ctx, out_buf| {
        float_dispatch::dispatch_float_into(&op, inputs, ctx, out_buf)
    })
}

/// Pre-compiled execution tape using boxed kernels.
///
/// Like [`Tape`] but uses [`BoxedInstruction`] (closures that capture op
/// parameters) instead of bare function pointers. Built by [`crate::tape_builder::build_tape`].
pub struct BoxedTape {
    /// Flat instruction array in execution order.
    pub instructions: Vec<BoxedInstruction>,
    /// Level boundaries: `level_offsets[i]..level_offsets[i+1]` is the range
    /// of instructions for level `i`.
    pub level_offsets: Vec<usize>,
}

impl BoxedTape {
    /// Create an empty boxed tape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            level_offsets: vec![0],
        }
    }

    /// Create a boxed tape with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(n_instructions: usize, n_levels: usize) -> Self {
        let mut level_offsets = Vec::with_capacity(n_levels + 1);
        level_offsets.push(0);
        Self {
            instructions: Vec::with_capacity(n_instructions),
            level_offsets,
        }
    }

    /// Add an instruction and return its index.
    pub fn push(&mut self, instr: BoxedInstruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Mark the end of the current level.
    pub fn end_level(&mut self) {
        self.level_offsets.push(self.instructions.len());
    }

    /// Number of levels in the tape.
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.level_offsets.len().saturating_sub(1)
    }

    /// Execute the boxed tape sequentially against the given arena.
    ///
    /// Uses swap-insert for zero-allocation buffer recycling after warmup.
    pub fn execute(
        &self,
        arena: &mut BufferArena<'_>,
        ctx: Option<&ExecutionContext>,
    ) -> ExecResult<()> {
        let mut out_buf: Vec<u8> = Vec::with_capacity(4096);

        for (i, instr) in self.instructions.iter().enumerate() {
            // Prefetch next instruction's input data.
            if i + 1 < self.instructions.len() {
                let next = &self.instructions[i + 1];
                for &idx in &next.input_indices {
                    let id = NodeId::new(idx, 0);
                    if let Ok(data) = arena.get(id) {
                        prefetch_read(data.as_ptr());
                    }
                }
            }

            // Gather input refs and execute kernel in a scoped block.
            {
                let input_refs: SmallVec<[&[u8]; 4]> = instr
                    .input_indices
                    .iter()
                    .map(|&idx| arena.get(NodeId::new(idx, 0)))
                    .collect::<ExecResult<SmallVec<_>>>()?;
                out_buf.clear();
                if instr.output_byte_hint > 0 {
                    out_buf.reserve(instr.output_byte_hint as usize);
                }
                (instr.kernel)(&input_refs, ctx, &mut out_buf)?;
            }

            let out_id = NodeId::new(instr.output_idx, 0);
            arena.swap_insert_with_elem_size(out_id, &mut out_buf, instr.output_elem_size as usize);
        }

        Ok(())
    }
}

impl Default for BoxedTape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_kernel(
        inputs: &[&[u8]],
        _ctx: Option<&ExecutionContext>,
        out: &mut Vec<u8>,
    ) -> ExecResult<()> {
        out.extend_from_slice(inputs[0]);
        Ok(())
    }

    #[test]
    fn empty_tape_executes() {
        let tape = Tape::new();
        let mut arena = BufferArena::new();
        assert!(tape.execute(&mut arena, None).is_ok());
    }

    #[test]
    fn single_instruction() {
        let mut tape = Tape::new();
        tape.push(Instruction {
            kernel: identity_kernel,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            tile_hint: 0,
        });
        tape.end_level();

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![1, 2, 3, 4]);

        tape.execute(&mut arena, None).unwrap();

        let out = arena.get(NodeId::new(1, 0)).unwrap();
        assert_eq!(out, &[1, 2, 3, 4]);
    }

    #[test]
    fn two_level_chain() {
        fn double_kernel(
            inputs: &[&[u8]],
            _ctx: Option<&ExecutionContext>,
            out: &mut Vec<u8>,
        ) -> ExecResult<()> {
            for &b in inputs[0] {
                out.push(b.wrapping_mul(2));
            }
            Ok(())
        }

        let mut tape = Tape::new();
        tape.push(Instruction {
            kernel: double_kernel,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 1,
            tile_hint: 0,
        });
        tape.end_level();
        tape.push(Instruction {
            kernel: double_kernel,
            output_idx: 2,
            input_indices: vec![1],
            output_elem_size: 1,
            tile_hint: 0,
        });
        tape.end_level();

        assert_eq!(tape.n_levels(), 2);

        let mut arena = BufferArena::new();
        arena.insert(NodeId::new(0, 0), vec![5]);

        tape.execute(&mut arena, None).unwrap();

        let out = arena.get(NodeId::new(2, 0)).unwrap();
        assert_eq!(out, &[20]); // 5 * 2 * 2
    }

    #[test]
    fn level_offsets_correct() {
        let mut tape = Tape::with_capacity(4, 2);
        tape.push(Instruction {
            kernel: identity_kernel,
            output_idx: 1,
            input_indices: vec![0],
            output_elem_size: 4,
            tile_hint: 0,
        });
        tape.push(Instruction {
            kernel: identity_kernel,
            output_idx: 2,
            input_indices: vec![0],
            output_elem_size: 4,
            tile_hint: 0,
        });
        tape.end_level();
        tape.push(Instruction {
            kernel: identity_kernel,
            output_idx: 3,
            input_indices: vec![1],
            output_elem_size: 4,
            tile_hint: 0,
        });
        tape.end_level();

        assert_eq!(tape.n_levels(), 2);
        assert_eq!(tape.level_offsets, vec![0, 2, 3]);
    }
}
