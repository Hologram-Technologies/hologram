//! Embedder-provided wasm worker pool for decode parallelism (plan 077
//! item 5).
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
//! **Determinism is structural.** A job partitions the GEMV's output rows
//! into contiguous per-participant ranges; every output row is computed by
//! exactly one participant running the identical single-threaded inner, so
//! the per-output reduction order — and therefore the output bits and every
//! CE derivation key — is unchanged from the serial path. The
//! `parallel_gemv_matches_serial_bitwise` test locks this.
//!
//! Embedder contract (fail-loud where checkable):
//! - Register every worker (call [`hologram_worker_run`]) **before the
//!   first execute**; registration after the first job traps.
//! - Call [`InferenceSession::execute`] from a worker, not the browser main
//!   thread (the join may block on `memory.atomic.wait32`).
//! - The workers never allocate; the embedder's global allocator only needs
//!   to serve the executing thread.
//! - no_std builds import `hologram_host_wait32` / `hologram_host_notify`
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
    fn hologram_host_wait32(ptr: *const i32, expect: i32, timeout_ns: i64) -> i32;
    /// Wake up to `count` waiters on `ptr`.
    fn hologram_host_notify(ptr: *const i32, count: u32) -> u32;
}

/// Width of the raw job slot. The slot is `usize` atomics because it lives in
/// shared linear memory and is published/drained across threads; [`GemvJob`] is
/// the typed form every caller and the executor actually use, and
/// [`GemvJob::encode`] / [`GemvJob::decode`] are the only code that knows which
/// word is which.
pub(crate) const JOB_ARGS: usize = 10;

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

/// One fork-join GEMV job: a `[n, k]` output-major weight times one quantized
/// activation row, sliced across participants by output row.
#[derive(Clone, Copy)]
pub(crate) struct GemvJob {
    /// The i8-quantized activation row, `k` entries.
    pub q: *const i8,
    /// Per-output-channel scales, `n` entries.
    pub scales: *const f32,
    /// Output row, `n` f32.
    pub out: *mut f32,
    pub k: usize,
    pub n: usize,
    /// The activation's dynamic scale (`amax / 127`).
    pub scale_a: f32,
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
        [
            self.q as usize,
            aux1,
            aux2,
            bq,
            self.scales as usize,
            self.out as usize,
            self.k,
            self.n,
            self.scale_a.to_bits() as usize,
            tag,
        ]
    }

    /// # Safety
    /// `args` must be the output of [`Self::encode`] for a job whose buffers are
    /// still live. An unknown tag is a corrupt slot, not a recoverable input.
    pub(crate) unsafe fn decode(args: &[usize; JOB_ARGS]) -> Self {
        let operands = match args[9] {
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
            scale_a: f32::from_bits(args[8] as u32),
            operands,
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

/// Below this weight size (`k·n` int8 bytes) a GEMV runs serial: the wake +
/// join round-trip (~µs) only pays once the per-participant slice is
/// meaningful. Structural (latency vs. work), not model-derived; decode
/// projections sit orders of magnitude above it.
const POOL_MIN_WEIGHT_BYTES: usize = 1 << 18;

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
        hologram_host_wait32(a.as_ptr() as *const i32, expect as i32, timeout_ns);
    }
}

#[inline]
fn notify_all(a: &AtomicU32) {
    #[cfg(feature = "std")]
    let _ = a; // spinners re-check; nothing to wake
    #[cfg(not(feature = "std"))]
    // SAFETY: as above; waking more waiters than exist is a no-op.
    unsafe {
        hologram_host_notify(a.as_ptr() as *const i32, u32::MAX);
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
        // return from `fork_join_gemv` until `done` reaches the snapshot),
        // ranges are disjoint per participant, and this worker's id is
        // below the snapshot by the registration contract. `args` was written
        // by `GemvJob::encode` before the epoch bump this thread observed.
        unsafe {
            let job = GemvJob::decode(&args);
            crate::cpu::simd::pool_exec_gemv(&job, worker_id as usize, parts);
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
pub(crate) fn fork_join_gemv(job: GemvJob) -> bool {
    let w = WORKERS.load(Ordering::Acquire);
    if w == 0 || job.k * job.n < POOL_MIN_WEIGHT_BYTES {
        return false;
    }
    let args = job.encode();
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
        crate::cpu::simd::pool_exec_gemv(&job, parts as usize - 1, parts as usize);
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
