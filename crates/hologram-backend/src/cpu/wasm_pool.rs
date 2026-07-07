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

/// Raw job arguments: `[q, qp, qn, bq, scales, out, k, n, scale_a_bits]`.
pub(crate) const JOB_ARGS: usize = 9;

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
        // below the snapshot by the registration contract.
        unsafe {
            crate::cpu::simd::pool_exec_gemv(&args, worker_id as usize, parts);
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
pub(crate) fn fork_join_gemv(args: [usize; JOB_ARGS]) -> bool {
    let w = WORKERS.load(Ordering::Acquire);
    let (k, n) = (args[6], args[7]);
    if w == 0 || k * n < POOL_MIN_WEIGHT_BYTES {
        return false;
    }
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
        crate::cpu::simd::pool_exec_gemv(&args, parts as usize - 1, parts as usize);
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
