# Plan: Multi-Backend Dispatch Architecture

## Context

Hologram's tape executor currently dispatches to CPU kernels with compile-time feature gating (`accelerate` for macOS BLAS, `simd` for NEON/AVX2, `parallel` for Rayon). The Phase 14 monomorphized dispatch showed 8.5x speedup from enabling autovectorization — but this only scratches the surface for GPU-capable hardware.

This plan formalizes the backend abstraction to support: **CPU** (current, with SIMD tiers), **Metal** (Apple GPU), **CUDA** (NVIDIA GPU), and **WebGPU** (browser via wgpu). Backend selection is compile-time via feature flags — zero runtime overhead, matching the existing pattern.

**Scope for this sprint**: Formalize the backend abstraction with auto-detection + runtime selection + CPU implementation. GPU stubs compile but fall back to CPU. Actual Metal/CUDA/WebGPU kernel implementations are future sprints.

## Design Decisions

### Auto-Detection (build.rs)
Instead of requiring `--features metal`, `build.rs` probes the build machine:
- **Metal**: check `target_os = "macos"` (Metal is always available on macOS 10.14+)
- **CUDA**: check for `nvcc` on PATH or `CUDA_HOME` env var
- **WebGPU**: check `target_arch = "wasm32"`
- **CPU**: always available

`build.rs` emits `cargo:rustc-cfg=has_metal`, `cargo:rustc-cfg=has_cuda`, etc. Code uses `#[cfg(has_metal)]` instead of `#[cfg(feature = "metal")]`. Feature flags still exist as manual overrides (`--features force_metal`).

### Runtime Selection
`TapeContext` carries a `BackendSelector` that picks among compiled-in backends:
```rust
pub enum BackendSelector {
    Auto,           // Best available (GPU → CPU)
    Cpu,            // Force CPU even if GPU available
    Metal,          // Force Metal (error if not compiled in)
    Cuda,           // Force CUDA
    WebGpu,         // Force WebGPU
}
```

`dispatch_kernel` resolves this to a concrete `&dyn ComputeBackend` once at tape execution start (not per-instruction). The match cost is paid once, not per-op.

## Design

### Backend Trait

```rust
// crates/hologram-exec/src/backend/mod.rs

/// Compute backend for tape kernel dispatch.
///
/// Each backend implements dispatch for the op types it supports.
/// Unsupported ops return `None`, causing fallback to the CPU backend.
pub trait ComputeBackend {
    /// Dispatch a float op. Returns `Ok(true)` if handled.
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool>;

    /// Dispatch a matmul. Returns `Ok(true)` if handled.
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize, k: usize, n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool>;

    /// Name for diagnostics.
    fn name(&self) -> &'static str;
}
```

### Backend Implementations (module structure)

```
crates/hologram-exec/src/backend/
├── mod.rs          # ComputeBackend trait + CpuBackend
├── cpu.rs          # CpuBackend (monomorphized SIMD dispatch)
├── metal.rs        # #[cfg(feature = "metal")] MetalBackend stub
├── cuda.rs         # #[cfg(feature = "cuda")] CudaBackend stub
└── webgpu.rs       # #[cfg(feature = "webgpu")] WebGpuBackend stub
```

### CpuBackend

The current `dispatch_kernel` + `dispatch_float_into` + monomorphized unary dispatch logic moves into `CpuBackend`. This is a pure refactor — same code, organized behind the trait.

```rust
// backend/cpu.rs
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn dispatch_float(&self, op: &FloatOp, inputs: &[&[u8]], out_buf: &mut Vec<u8>) -> ExecResult<bool> {
        // Current monomorphized dispatch from dispatch_float_into
        Ok(true)
    }
    fn dispatch_matmul(&self, inputs: &[&[u8]], m: usize, k: usize, n: usize, out_buf: &mut Vec<u8>) -> ExecResult<bool> {
        // Current matmul dispatch (with Accelerate BLAS on macOS)
        Ok(true)
    }
    fn name(&self) -> &'static str { "cpu" }
}
```

### GPU Backend Stubs

```rust
// backend/metal.rs
#[cfg(feature = "metal")]
pub struct MetalBackend {
    // Will hold: MTLDevice, MTLCommandQueue, compiled pipelines
}

#[cfg(feature = "metal")]
impl ComputeBackend for MetalBackend {
    fn dispatch_float(&self, _op: &FloatOp, _inputs: &[&[u8]], _out_buf: &mut Vec<u8>) -> ExecResult<bool> {
        Ok(false) // Not yet implemented — fall back to CPU
    }
    // ...
    fn name(&self) -> &'static str { "metal" }
}
```

### Integration with TapeContext

```rust
// tape.rs — TapeContext gains backend selection
pub struct TapeContext<'a> {
    pub ctx: Option<ExecutionContext>,
    pub constants: &'a ConstantStore,
    pub weights: &'a [u8],
    pub weight_cache: RefCell<WeightCache>,
    pub kv_state: Option<RefCell<KvCacheState>>,
    pub backend: BackendSelector,  // NEW — resolved to &dyn at execute start
}
```

### BackendSelector Resolution

```rust
// backend/mod.rs
impl BackendSelector {
    /// Resolve to the best concrete backend for this build + selector.
    pub fn resolve(&self) -> Box<dyn ComputeBackend> {
        match self {
            Self::Auto => default_backend(),
            Self::Cpu => Box::new(cpu::CpuBackend),
            #[cfg(has_metal)]
            Self::Metal => Box::new(metal::MetalBackend::new()),
            #[cfg(has_cuda)]
            Self::Cuda => Box::new(cuda::CudaBackend::new()),
            #[cfg(has_webgpu)]
            Self::WebGpu => Box::new(webgpu::WebGpuBackend::new()),
            // If requested backend not compiled in, fall back to CPU.
            _ => Box::new(cpu::CpuBackend),
        }
    }
}
```

### dispatch_kernel Change

```rust
fn dispatch_kernel(
    kernel: &TapeKernel,
    inputs: &[&[u8]],
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    match kernel {
        TapeKernel::Float(op) => {
            // Try the selected backend first.
            if tape_ctx.backend.dispatch_float(op, inputs, out_buf)? {
                return Ok(());
            }
            // Fallback to CPU.
            float_dispatch::dispatch_float_into(op, inputs, tape_ctx.ctx.as_ref(), out_buf)
        }
        TapeKernel::MatMulLut4(cid) => {
            // LUT-GEMM is CPU-only (quantized weights are CPU-specific).
            dispatch_lut_gemm_4(inputs, *cid, tape_ctx, out_buf)
        }
        // ... rest unchanged
    }
}
```

### Build-Time Auto-Detection (build.rs)

```rust
// crates/hologram-exec/build.rs — extended
fn main() {
    // Existing: link Accelerate framework on macOS
    #[cfg(feature = "accelerate")]
    { /* existing code */ }

    // Auto-detect Metal (macOS 10.14+ always has Metal)
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-cfg=has_metal");
    }

    // Auto-detect CUDA (check for nvcc or CUDA_HOME)
    if std::env::var("CUDA_HOME").is_ok()
        || std::process::Command::new("nvcc").arg("--version").output().is_ok()
    {
        println!("cargo:rustc-cfg=has_cuda");
    }

    // Auto-detect WebGPU (wasm32 target)
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32" {
        println!("cargo:rustc-cfg=has_webgpu");
    }
}
```

### Feature Flags (manual overrides)

```toml
# crates/hologram-exec/Cargo.toml
[features]
force_metal = []   # Force Metal even if auto-detection fails
force_cuda = []    # Force CUDA
force_webgpu = []  # Force WebGPU
```

### Default Backend Selection

```rust
// backend/mod.rs
/// Returns the best available backend for the current build.
/// Priority: CUDA > Metal > WebGPU > CPU
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(has_cuda)]
    { return Box::new(cuda::CudaBackend::new()); }

    #[cfg(has_metal)]
    { return Box::new(metal::MetalBackend::new()); }

    #[cfg(has_webgpu)]
    { return Box::new(webgpu::WebGpuBackend::new()); }

    Box::new(cpu::CpuBackend)
}

/// List all backends available in this build.
pub fn available_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    #[cfg(has_metal)] v.push("metal");
    #[cfg(has_cuda)] v.push("cuda");
    #[cfg(has_webgpu)] v.push("webgpu");
    v
}
```

## Step-by-Step Implementation

### Step 1: Create backend module skeleton
- New directory: `crates/hologram-exec/src/backend/`
- `mod.rs`: `ComputeBackend` trait definition
- `cpu.rs`: `CpuBackend` struct implementing the trait (wraps current dispatch logic)
- `metal.rs`, `cuda.rs`, `webgpu.rs`: stubs that return `Ok(false)` for everything

### Step 2: Add feature flags
- `crates/hologram-exec/Cargo.toml`: add `metal = []`, `cuda = []`, `webgpu = []` features
- Workspace `Cargo.toml`: add `metal`, `cuda`, `webgpu` features forwarding to hologram-exec

### Step 3: Wire CpuBackend into TapeContext
- Add `backend: &'a dyn ComputeBackend` to `TapeContext`
- Update `TapeContext::new` to default to `CpuBackend`
- Update `execute_tape` in `mmap/mod.rs`

### Step 4: Update dispatch_kernel to use backend
- Try `tape_ctx.backend.dispatch_float()` first
- Fall back to existing CPU dispatch

### Step 5: Update SPRINT.md
- Add Sprint 16 planning section

### Step 6: Tests
- Verify CpuBackend produces identical results to current dispatch
- Verify `default_backend()` returns CpuBackend when no GPU features enabled

## Files to Modify/Create

| File | Change |
|------|--------|
| `crates/hologram-exec/src/backend/mod.rs` | **NEW** — ComputeBackend trait + default_backend() |
| `crates/hologram-exec/src/backend/cpu.rs` | **NEW** — CpuBackend wrapping current dispatch |
| `crates/hologram-exec/src/backend/metal.rs` | **NEW** — stub |
| `crates/hologram-exec/src/backend/cuda.rs` | **NEW** — stub |
| `crates/hologram-exec/src/backend/webgpu.rs` | **NEW** — stub |
| `crates/hologram-exec/src/lib.rs` | Add `pub mod backend;` |
| `crates/hologram-exec/src/tape.rs` | Add `backend` field to TapeContext |
| `crates/hologram-exec/src/mmap/mod.rs` | Pass backend to TapeContext |
| `crates/hologram-exec/Cargo.toml` | Add feature flags |
| `Cargo.toml` | Forward feature flags |
| `specs/SPRINT.md` | Add Sprint 16 |
| `specs/plans/009-multi-backend-dispatch.md` | This plan |

## What Does NOT Change

- All existing CPU kernel code — stays in float_dispatch/, lut_gemm/, etc.
- The `TapeKernel` enum — backend dispatch is orthogonal to kernel type
- KvExecutor path — unaffected
- Archive format, graph format — unchanged
- Existing feature flags (accelerate, simd, parallel) — unchanged

## Future Sprints (not this PR)

- **Sprint 17**: Metal compute shader kernels for matmul, elementwise, softmax
- **Sprint 18**: CUDA kernel implementations
- **Sprint 19**: WebGPU/wgpu compute shader path
- **Sprint 20**: GPU memory management (buffer pooling, async transfers)

## Verification

- `cargo test --workspace` — all tests pass (CpuBackend default)
- `cargo clippy --workspace -- -D warnings` — zero warnings
- `cargo check --workspace` — auto-detects Metal on macOS, compiles stubs
- `just bench` — no regression (CpuBackend produces identical code paths)
- `available_backends()` returns `["cpu", "metal"]` on macOS, `["cpu"]` on Linux
- `BackendSelector::Auto` resolves to Metal on macOS, CPU on Linux
- `BackendSelector::Cpu` forces CPU even on Metal-capable machines
