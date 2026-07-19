//! Embedder-provided wasm worker pool for **inference** parallelism — the
//! decode GEMV (`m == 1`) and the prefill / batched-verify GEMM (`m > 1`)
//! through one output-column partition (plan 077 item 5; prefill pooling per
//! hologram-ai ADR-0018's upstream request: TTFT is the serial m>1 GEMM, and
//! the embedder cannot fix it — `m` pooled single-row jobs reload the weight
//! `m` times and lose to serial batched, so the batched kernel itself is what
//! must be partitioned).
//!
//! wasm32 has no `std::thread`: parallelism comes from the **embedder**
//! (hologram-ai serves its own COOP/COEP headers) instantiating this module
//! on N web workers that share one linear memory (a `+atomics,+bulk-memory`
//! shared-memory build), each calling the exported
//! [`hologram_worker_run`] once. Work then flows through a single job slot
//! in linear memory — published by the executing thread, drained by the
//! workers via `memory.atomic.wait32`/`notify` — as a flat fork-join: no
//! nested spawns, no queue growth, no allocation on any worker.
//!
//! **Determinism is structural.** A job partitions the output **columns**
//! into contiguous per-participant ranges; every output cell (all `m` rows of
//! a participant's columns) is computed by exactly one participant running
//! the identical single-threaded inner, so the per-output reduction order —
//! and therefore the output bits and every CE derivation key — is unchanged
//! from the serial path, at every `m`. The
//! `parallel_gemv_matches_serial_bitwise` test locks this for i8, packed-i4
//! and E8CB at `m ∈ {1, 2, 5, 8}`.
//!
//! Embedder contract (fail-loud where checkable):
//! - Register every worker (call [`hologram_worker_run`]) **before the
//!   first execute**; registration after the first job traps.
//! - Call [`InferenceSession::execute`] from a worker, not the browser main
//!   thread (the join may block on `memory.atomic.wait32`).
//! - The workers never allocate; the embedder's global allocator only needs
//!   to serve the executing thread.
//! - no_std builds import `hologram_types_wait32` / `hologram_types_notify`
//!   (JS `Atomics.wait` / `Atomics.notify` over the shared memory).
//! - [`hologram_pool_shutdown`] releases the workers (they return);
//!   [`hologram_pool_workers`] reports the registered count.
//!
//! Compiled only for `wasm32 + atomics + simd128`; the plain simd128 build
//! (no atomics) is byte-identical to before this module existed and remains
//! the witnessed single-threaded fallback.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

// The futex is embedder-provided, like the workers themselves: wasm's
// native `memory.atomic.wait32`/`notify` intrinsics are unstable on stable
// Rust (rust-lang/rust#77839), and the embedder already owns the worker
// lifecycle. In the browser these are two one-line JS imports over the
// shared memory (`Atomics.wait` / `Atomics.notify` on an Int32Array view);
// they are only imported by `wasm-threads` no_std builds. The `std` build
// (the wasmtime test lane, wasm32-wasip1-threads) parks by spin + OS yield
// instead — the synchronization algebra (epoch/done/shutdown atomics) is
// identical, only the idle mechanics differ.
#[cfg(not(feature = "std"))]
extern "C" {
    /// Block while `*ptr == expect`, up to `timeout_ns` (< 0 = infinite).
    fn hologram_types_wait32(ptr: *const i32, expect: i32, timeout_ns: i64) -> i32;
    /// Wake up to `count` waiters on `ptr`.
    fn hologram_types_notify(ptr: *const i32, count: u32) -> u32;
}

/// Width of the raw job slot. The slot is `usize` atomics because it lives in
/// shared linear memory and is published/drained across threads; [`GemvJob`] is
/// the typed form every caller and the executor actually use, and
/// [`GemvJob::encode`] / [`GemvJob::decode`] are the only code that knows which
/// word is which.
pub(crate) const JOB_ARGS: usize = 17;

/// The weight encoding a pooled GEMV runs, with its operands named.
///
/// Each variant fixes its own weight stride (bytes per output row) and its own
/// auxiliary operand; there is no shared "aux" word whose meaning depends on a
/// tag read elsewhere.
#[derive(Clone, Copy)]
pub(crate) enum GemvOperands {
    /// int8 output-major weight, `k` bytes per output row. `qp`/`qn` are the
    /// relaxed-SIMD split of the activation into non-negative halves; they are
    /// null on builds without `relaxed-simd`, where the kernel does not read them.
    I8 {
        bq: *const i8,
        qp: *const i8,
        qn: *const i8,
    },
    /// Packed int4 weight, `k/2` bytes per output row. `de` is the
    /// de-interleaved activation layout `matmul_i4_pc_omajor` builds.
    I4 { bq: *const u8, de: *const i8 },
    /// E8 codebook indices, `k/8` bytes per output row. `codebook` is the
    /// model's own codebook, pre-widened to i16.
    E8cb { bq: *const u8, codebook: *const i16 },
}

impl GemvOperands {
    /// Discriminant stored in the job slot. Explicit, because it crosses a
    /// shared-memory boundary — never `as usize` on the enum.
    const TAG_I8: usize = 0;
    const TAG_I4: usize = 1;
    const TAG_E8CB: usize = 2;
}

/// One fork-join job: a `[n, k]` output-major weight times `m` quantized
/// activation rows, sliced across participants by **output column**.
///
/// `m == 1` is the decode GEMV. `m > 1` is the prefill / batched-verify GEMM:
/// each participant computes its contiguous column range for **all** `m` rows,
/// reading its weight-column tile once — the batched form the serial kernel
/// already uses, partitioned. Doing prefill as `m` pooled single-row jobs would
/// reload the whole weight `m` times and lose to serial batched; the batched
/// pooled job is what actually beats it.
#[derive(Clone, Copy)]
pub(crate) struct GemvJob {
    /// The i8-quantized activation rows, `m · k` entries (row `i` at `i·k`).
    pub q: *const i8,
    /// Per-output-channel scales, `n` entries.
    pub scales: *const f32,
    /// Output, `m · n` f32 (row `i` at `i·n`).
    pub out: *mut f32,
    pub k: usize,
    pub n: usize,
    /// Row count. 1 = decode GEMV; >1 = prefill GEMM.
    pub m: usize,
    /// Per-row activation scales (`amax_i / 127`), `m` entries. Each row is
    /// quantized per token, so each carries its own scale; a row whose scale is
    /// `0.0` was all-zero and its output row is pre-zeroed by the publisher —
    /// participants skip it.
    pub sa: *const f32,
    pub operands: GemvOperands,
}

impl GemvJob {
    pub(crate) fn encode(&self) -> [usize; JOB_ARGS] {
        let (aux1, aux2, bq, tag) = match self.operands {
            GemvOperands::I8 { bq, qp, qn } => {
                (qp as usize, qn as usize, bq as usize, GemvOperands::TAG_I8)
            }
            GemvOperands::I4 { bq, de } => (de as usize, 0, bq as usize, GemvOperands::TAG_I4),
            GemvOperands::E8cb { bq, codebook } => {
                (codebook as usize, 0, bq as usize, GemvOperands::TAG_E8CB)
            }
        };
        // The job-kind tag lives at the fixed last word (`JOB_ARGS - 1`) for
        // every job type; unused middle words are zero.
        let mut a = [0usize; JOB_ARGS];
        a[0] = self.q as usize;
        a[1] = aux1;
        a[2] = aux2;
        a[3] = bq;
        a[4] = self.scales as usize;
        a[5] = self.out as usize;
        a[6] = self.k;
        a[7] = self.n;
        a[8] = self.sa as usize;
        a[9] = self.m;
        a[JOB_ARGS - 1] = tag;
        a
    }

    /// # Safety
    /// `args` must be the output of [`Self::encode`] for a job whose buffers are
    /// still live. An unknown tag is a corrupt slot, not a recoverable input.
    pub(crate) unsafe fn decode(args: &[usize; JOB_ARGS]) -> Self {
        let operands = match args[JOB_ARGS - 1] {
            GemvOperands::TAG_I8 => GemvOperands::I8 {
                bq: args[3] as *const i8,
                qp: args[1] as *const i8,
                qn: args[2] as *const i8,
            },
            GemvOperands::TAG_I4 => GemvOperands::I4 {
                bq: args[3] as *const u8,
                de: args[1] as *const i8,
            },
            GemvOperands::TAG_E8CB => GemvOperands::E8cb {
                bq: args[3] as *const u8,
                codebook: args[1] as *const i16,
            },
            other => panic!("wasm_pool: corrupt job slot, unknown GEMV tag {other}"),
        };
        Self {
            q: args[0] as *const i8,
            scales: args[4] as *const f32,
            out: args[5] as *mut f32,
            k: args[6],
            n: args[7],
            sa: args[8] as *const f32,
            m: args[9],
            operands,
        }
    }
}

/// One fork-join **decode attention** job: `b·h·m` independent
/// `(batch, head, query-row)` outputs over `past ∥ new` keys, sliced across
/// participants by **row**. Each row is a whole score→softmax→context
/// pipeline, so the partition is bit-identical to serial by construction.
///
/// `scores` is publisher-allocated scratch of `participants · (past + new)`
/// f32; participant `p` uses its own stripe — workers never allocate (the
/// embedder contract), so the publisher carries the scratch for everyone.
#[derive(Clone, Copy)]
pub(crate) struct AttnJob {
    pub q: *const f32,
    pub k_past: *const f32,
    pub v_past: *const f32,
    pub k_new: *const f32,
    pub v_new: *const f32,
    pub mask: *const f32,
    pub out: *mut f32,
    pub scores: *mut f32,
    pub b: usize,
    pub h: usize,
    pub hkv: usize,
    pub m: usize,
    pub past: usize,
    pub new: usize,
    pub d: usize,
    /// `f32::to_bits` of the score divisor (already resolved by the caller).
    pub scale_bits: u32,
}

/// Slot tag for an attention job (GEMV tags are 0..=2).
pub(crate) const TAG_ATTN: usize = 3;

impl AttnJob {
    pub(crate) fn encode(&self) -> [usize; JOB_ARGS] {
        [
            self.q as usize,
            self.k_past as usize,
            self.v_past as usize,
            self.k_new as usize,
            self.v_new as usize,
            self.mask as usize,
            self.out as usize,
            self.scores as usize,
            self.b,
            self.h,
            self.hkv,
            self.m,
            self.past,
            self.new,
            self.d,
            self.scale_bits as usize,
            TAG_ATTN, // JOB_ARGS - 1: the job-kind tag
        ]
    }

    /// # Safety
    /// `args` must be the output of [`Self::encode`] for a live job.
    pub(crate) unsafe fn decode(args: &[usize; JOB_ARGS]) -> Self {
        Self {
            q: args[0] as *const f32,
            k_past: args[1] as *const f32,
            v_past: args[2] as *const f32,
            k_new: args[3] as *const f32,
            v_new: args[4] as *const f32,
            mask: args[5] as *const f32,
            out: args[6] as *mut f32,
            scores: args[7] as *mut f32,
            b: args[8],
            h: args[9],
            hkv: args[10],
            m: args[11],
            past: args[12],
            new: args[13],
            d: args[14],
            scale_bits: args[15] as u32,
        }
    }
}

/// One fork-join **scalar-mask decode attention** job (κ121): as
/// [`AttnJob`], but the visibility law is compiled in — no mask pointer;
/// `vis_past` (the realized past prefix, already clamped) rides where the
/// mask pointer sat, so the encoding still fits the fixed job slot.
/// `scores` is publisher-allocated scratch of
/// `participants · (vis_past + new)` f32, stripe per participant.
#[derive(Clone, Copy)]
pub(crate) struct AttnValidJob {
    pub q: *const f32,
    pub k_past: *const f32,
    pub v_past: *const f32,
    pub k_new: *const f32,
    pub v_new: *const f32,
    pub out: *mut f32,
    pub scores: *mut f32,
    pub b: usize,
    pub h: usize,
    pub hkv: usize,
    pub m: usize,
    pub past: usize,
    pub new: usize,
    pub d: usize,
    /// Realized past prefix, `≤ past`.
    pub vis_past: usize,
    /// `f32::to_bits` of the score divisor (already resolved by the caller).
    pub scale_bits: u32,
}

/// Slot tag for a scalar-mask attention job.
pub(crate) const TAG_ATTN_VALID: usize = 4;

impl AttnValidJob {
    pub(crate) fn encode(&self) -> [usize; JOB_ARGS] {
        [
            self.q as usize,
            self.k_past as usize,
            self.v_past as usize,
            self.k_new as usize,
            self.v_new as usize,
            self.vis_past,
            self.out as usize,
            self.scores as usize,
            self.b,
            self.h,
            self.hkv,
            self.m,
            self.past,
            self.new,
            self.d,
            self.scale_bits as usize,
            TAG_ATTN_VALID, // JOB_ARGS - 1: the job-kind tag
        ]
    }

    /// # Safety
    /// `args` must be the output of [`Self::encode`] for a live job.
    pub(crate) unsafe fn decode(args: &[usize; JOB_ARGS]) -> Self {
        Self {
            q: args[0] as *const f32,
            k_past: args[1] as *const f32,
            v_past: args[2] as *const f32,
            k_new: args[3] as *const f32,
            v_new: args[4] as *const f32,
            vis_past: args[5],
            out: args[6] as *mut f32,
            scores: args[7] as *mut f32,
            b: args[8],
            h: args[9],
            hkv: args[10],
            m: args[11],
            past: args[12],
            new: args[13],
            d: args[14],
            scale_bits: args[15] as u32,
        }
    }
}

struct Job {
    /// Bumped to publish a job; workers wait on it.
    epoch: AtomicU32,
    /// Participant count snapshot at publish (registered workers + main).
    participants: AtomicU32,
    /// Completion counter; the publisher waits for it to reach the worker
    /// count it snapshot.
    done: AtomicU32,
    /// Nonzero ⇒ workers return.
    shutdown: AtomicU32,
    args: [AtomicUsize; JOB_ARGS],
}

#[allow(clippy::declare_interior_mutable_const)]
const AZ: AtomicUsize = AtomicUsize::new(0);
static JOB: Job = Job {
    epoch: AtomicU32::new(0),
    participants: AtomicU32::new(0),
    done: AtomicU32::new(0),
    shutdown: AtomicU32::new(0),
    args: [AZ; JOB_ARGS],
};
static WORKERS: AtomicU32 = AtomicU32::new(0);

/// Below this many multiply-accumulates (`m·k·n`) a job runs serial: the wake +
/// join round-trip (~µs) only pays once the per-participant slice is
/// meaningful. Structural (latency vs. work), not model-derived.
///
/// The gate is **work**, not weight bytes. At `m == 1` the two coincide (one
/// MAC per weight byte), so decode behaves exactly as before. At `m > 1` a
/// byte-keyed floor is blind to the batch: a 112 KiB per-head projection at
/// `m = 128` is 14.7 MMAC of embarrassingly parallel work — measured 908 µs
/// serial under the old floor, with the pool idle. Prefill work scales with
/// `m`; the admission test must too.
const POOL_MIN_MACS: usize = 1 << 18;

/// Minimum output columns per participant for a job to be worth partitioning.
///
/// The column partition hands each participant `~n/parts` columns; the widest
/// SIMD inner consumes 8 at a time, so a narrower slice runs every row through
/// the scalar column tail. Measured at `128×1536×8` with 4 participants (2
/// columns each): pooled 473 µs vs serial 342 µs — the pool *loses* when the
/// slices are sub-SIMD-width, however much total work the job carries. Work
/// admits a job; width must too.
const POOL_MIN_COLS_PER_PARTICIPANT: usize = 8;

#[inline]
fn wait_u32(a: &AtomicU32, expect: u32, timeout_ns: i64) {
    #[cfg(feature = "std")]
    {
        // Test-lane park: re-check is the caller's loop; just cede the CPU.
        let _ = timeout_ns;
        if a.load(Ordering::Acquire) == expect {
            std::thread::yield_now();
        }
    }
    #[cfg(not(feature = "std"))]
    // SAFETY: the atomic lives in shared linear memory for the program's
    // lifetime; the embedder futex blocks only while `*ptr == expect`.
    unsafe {
        hologram_types_wait32(a.as_ptr() as *const i32, expect as i32, timeout_ns);
    }
}

#[inline]
fn notify_all(a: &AtomicU32) {
    #[cfg(feature = "std")]
    let _ = a; // spinners re-check; nothing to wake
    #[cfg(not(feature = "std"))]
    // SAFETY: as above; waking more waiters than exist is a no-op.
    unsafe {
        hologram_types_notify(a.as_ptr() as *const i32, u32::MAX);
    }
}

/// Worker entry: register, then drain jobs until shutdown. Exported for the
/// embedder; each spawned worker calls it exactly once and never returns
/// until [`hologram_pool_shutdown`].
#[no_mangle]
pub extern "C" fn hologram_worker_run(worker_id: u32) {
    // Fail loud on late registration: participant snapshots are only sound
    // if the worker set is fixed before the first publish.
    assert_eq!(
        JOB.epoch.load(Ordering::Acquire),
        0,
        "hologram_worker_run: workers must register before the first execute"
    );
    WORKERS.fetch_add(1, Ordering::SeqCst);
    let mut seen = 0u32;
    loop {
        if JOB.shutdown.load(Ordering::Acquire) != 0 {
            WORKERS.fetch_sub(1, Ordering::SeqCst);
            return;
        }
        let e = JOB.epoch.load(Ordering::Acquire);
        if e == seen {
            // 5 ms timeout so shutdown is always observed promptly.
            wait_u32(&JOB.epoch, seen, 5_000_000);
            continue;
        }
        seen = e;
        let parts = JOB.participants.load(Ordering::Acquire) as usize;
        let mut args = [0usize; JOB_ARGS];
        for (slot, a) in args.iter_mut().zip(JOB.args.iter()) {
            *slot = a.load(Ordering::Acquire);
        }
        // SAFETY: the publisher keeps the job's buffers alive (it does not
        // return from the fork-join until `done` reaches the snapshot),
        // ranges are disjoint per participant, and this worker's id is
        // below the snapshot by the registration contract. `args` was written
        // by the job's `encode` before the epoch bump this thread observed.
        unsafe {
            exec_slot(&args, worker_id as usize, parts);
        }
        JOB.done.fetch_add(1, Ordering::AcqRel);
        notify_all(&JOB.done);
    }
}

/// Registered worker count (embedder sanity/poll surface).
#[no_mangle]
pub extern "C" fn hologram_pool_workers() -> u32 {
    WORKERS.load(Ordering::Acquire)
}

/// Release the workers: they observe the flag (≤ 5 ms) and return.
#[no_mangle]
pub extern "C" fn hologram_pool_shutdown() {
    JOB.shutdown.store(1, Ordering::Release);
    notify_all(&JOB.epoch);
}

/// Fork-join one GEMV across the registered workers + the calling thread.
/// Returns `false` (caller runs serial) when no workers are registered or
/// the job is below the latency floor. On `true` the output rows are fully
/// written before returning.
/// Route one participant's share of the published job by its kind tag.
///
/// # Safety
/// `args` is a live job slot per the publisher contract; participant ranges
/// are disjoint.
unsafe fn exec_slot(args: &[usize; JOB_ARGS], part: usize, parts: usize) {
    if args[JOB_ARGS - 1] == TAG_ATTN {
        let job = AttnJob::decode(args);
        crate::cpu::float_kernels::pool_exec_attn(&job, part, parts);
    } else if args[JOB_ARGS - 1] == TAG_ATTN_VALID {
        let job = AttnValidJob::decode(args);
        crate::cpu::float_kernels::pool_exec_attn_valid(&job, part, parts);
    } else {
        let job = GemvJob::decode(args);
        crate::cpu::simd::pool_exec_gemv(&job, part, parts);
    }
}

pub(crate) fn fork_join_gemv(job: GemvJob) -> bool {
    let w = WORKERS.load(Ordering::Acquire);
    if w == 0
        || job.m.saturating_mul(job.k).saturating_mul(job.n) < POOL_MIN_MACS
        || job.n < POOL_MIN_COLS_PER_PARTICIPANT * (w as usize + 1)
    {
        return false;
    }
    fork_join_raw(job.encode(), w)
}

/// Registered workers + the calling thread — the participant count a pooled
/// job will be sliced across (1 ⇒ no pool). Publishers size per-participant
/// scratch with this; the worker set is frozen after the first publish, so the
/// value is stable across the publish it precedes.
pub(crate) fn participants() -> usize {
    WORKERS.load(Ordering::Acquire) as usize + 1
}

/// Fork-join one decode-attention job across the pool by **row**. Admission:
/// at least two rows (each participant computes whole rows) and enough MACs
/// (`rows · keys · head_dim`) to amortize the wake + join round-trip — the
/// same work-based floor the GEMV uses.
pub(crate) fn fork_join_attn(job: AttnJob) -> bool {
    let w = WORKERS.load(Ordering::Acquire);
    let rows = job.b * job.h * job.m;
    let l = job.past + job.new;
    if w == 0 || rows < 2 || rows.saturating_mul(l).saturating_mul(job.d) < POOL_MIN_MACS {
        return false;
    }
    fork_join_raw(job.encode(), w)
}

/// Fork-join one scalar-mask decode-attention job by **row**. Same
/// admission as the mask form, but the work term is the **effective**
/// visible width (`vis_past + new`) — the kernel only reads that many
/// columns per row, so a barely-realized 32K bucket correctly declines.
pub(crate) fn fork_join_attn_valid(job: AttnValidJob) -> bool {
    let w = WORKERS.load(Ordering::Acquire);
    let rows = job.b * job.h * job.m;
    let l_vis = job.vis_past + job.new;
    if w == 0 || rows < 2 || rows.saturating_mul(l_vis).saturating_mul(job.d) < POOL_MIN_MACS {
        return false;
    }
    fork_join_raw(job.encode(), w)
}

/// Publish an encoded job, run the calling thread's share, join.
fn fork_join_raw(args: [usize; JOB_ARGS], w: u32) -> bool {
    let parts = w + 1;
    for (a, v) in JOB.args.iter().zip(args.iter()) {
        a.store(*v, Ordering::Release);
    }
    JOB.participants.store(parts, Ordering::Release);
    JOB.done.store(0, Ordering::Release);
    JOB.epoch.fetch_add(1, Ordering::AcqRel);
    notify_all(&JOB.epoch);
    // The calling thread is the last participant.
    // SAFETY: args point into live workspace buffers owned by the caller;
    // this range is disjoint from every worker's.
    unsafe {
        exec_slot(&args, parts as usize - 1, parts as usize);
    }
    // Join: brief spin, then park on `done`. The Acquire loads pair with
    // the workers' AcqRel increments, making their row writes visible.
    let mut spins = 0u32;
    loop {
        let d = JOB.done.load(Ordering::Acquire);
        if d >= w {
            return true;
        }
        spins = spins.wrapping_add(1);
        if spins < 256 {
            core::hint::spin_loop();
        } else {
            wait_u32(&JOB.done, d, 1_000_000);
        }
    }
}
