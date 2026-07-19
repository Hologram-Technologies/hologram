//! In-tree persistent worker pool for UOR-native multi-core execution.
//!
//! Parallelism in hologram is **not** a bolted-on layer: it is the
//! cache-oblivious lattice recursion read as a task DAG. The matmul recursion
//! bisects the largest of m/n/k; an M- or N-split yields two **disjoint-output,
//! independent** sub-products (lattice nodes). This module cuts that recursion
//! tree at the *parallel grain* — bisecting the output into ≈one tile per core
//! ([`output_tiles`]) — and runs the frontier across a persistent pool, each
//! tile then executing the **sequential** cache-oblivious recursion. The same
//! tree that gives single-core cache-obliviousness gives the parallel tasks;
//! per-core private cache is what makes a bandwidth-bound problem compound past
//! linear (each tile's working set fits one core's L2).
//!
//! The pool is `std`-only (gated behind `parallel`), built on `std::thread`
//! with no external dependency. Tasks are flat (a tile never spawns more
//! tasks), so the calling thread participates as a worker and there is no
//! nested-join deadlock. The single-thread path is unaffected when the feature
//! is off.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;

type Task = Box<dyn FnOnce() + Send>;

/// Bounded busy-wait a worker runs before parking on the condvar. Decode
/// submits ~one GEMV fork-join per projection, back-to-back and microseconds
/// apart; keeping workers hot across that gap turns each barrier into a
/// lock-free queue poll instead of a futex park/unpark of the whole pool (the
/// dominant cost when the per-op work is small). `spin_loop` issues `PAUSE`, so
/// a spinning worker yields its SMT sibling's pipeline rather than fighting it.
const WORKER_SPIN: u32 = 1 << 14;

/// Decrements the run's completion counter on scope exit — including an
/// unwinding `t()`. Without it a panicking task would never decrement and the
/// caller's barrier would spin forever; with it the barrier always completes
/// (a panicking worker still dies, but the pool cannot deadlock).
struct Countdown<'a>(&'a AtomicUsize);
impl Drop for Countdown<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

struct Shared {
    queue: Mutex<VecDeque<Task>>,
    cv: Condvar,
    /// Lock-free queue-length hint: a worker only takes the queue mutex when
    /// this reads > 0, so idle spinning never touches the lock. Kept in sync
    /// under the queue lock on every push, and on every pop (by whoever pops).
    pending: AtomicUsize,
}

/// A persistent set of worker threads sharing one task queue.
pub struct Pool {
    shared: Arc<Shared>,
    width: usize,
    /// Admits **one batch at a time** to the shared queue. Without it, two
    /// hosts walking concurrently interleave their batches, and the caller
    /// help-drain in [`Pool::run`] executes *foreign* tasks — under a held
    /// thread-local scratch borrow that a foreign task may re-enter
    /// (`RefCell already borrowed`), with the unwind then orphaning the rest
    /// of that batch in the queue: raw operand pointers outliving their walk
    /// (the v0.9.0 concurrent-session unsoundness). Serializing batches makes
    /// concurrent hosts exactly as correct as sequential ones; the pool is
    /// width-bounded, so overlapping batches could not add throughput anyway.
    run_lock: Mutex<()>,
}

/// Batch-scoped cleanup, run on every exit from [`Pool::run`] — normal or
/// unwinding. Cancels whatever is still queued (with the batch lock held the
/// queue holds only this batch's tasks) and then waits for every in-flight
/// task to finish, so **no task ever outlives its `run` call**: the raw
/// operand pointers a task captures must never dangle into a finished (or
/// panicked and dropped) walk. A cancelled wrapper never constructs its
/// `Countdown`, so the barrier target is the cancellation count.
struct BatchDrain<'a> {
    shared: &'a Shared,
    remaining: &'a AtomicUsize,
}

impl Drop for BatchDrain<'_> {
    fn drop(&mut self) {
        let mut cancelled = 0usize;
        loop {
            let t = {
                let mut q = self.shared.queue.lock().unwrap();
                let t = q.pop_front();
                if t.is_some() {
                    self.shared.pending.fetch_sub(1, Ordering::Relaxed);
                }
                t
            };
            match t {
                Some(task) => {
                    drop(task);
                    cancelled += 1;
                }
                None => break,
            }
        }
        let mut spins = 0u32;
        while self.remaining.load(Ordering::Acquire) != cancelled {
            if spins < WORKER_SPIN {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// The process-wide pool, lazily sized to the available parallelism.
pub fn pool() -> &'static Pool {
    POOL.get_or_init(Pool::new)
}

impl Pool {
    fn new() -> Self {
        let width = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            pending: AtomicUsize::new(0),
        });
        // `width - 1` persistent workers; the submitting thread is the width-th
        // runner (it drains the queue in `run`), so a 1-core host spawns none.
        for _ in 1..width {
            let sh = Arc::clone(&shared);
            thread::spawn(move || worker_loop(&sh));
        }
        Pool {
            shared,
            width,
            run_lock: Mutex::new(()),
        }
    }

    /// Number of concurrent runners (cores).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Run `tasks` to completion. The calling thread participates (drains the
    /// queue), so flat task sets never deadlock regardless of `width`. Returns
    /// only after every task has finished.
    ///
    /// Completion is an `AtomicUsize` the caller spins on: each task
    /// `fetch_sub`s (Release), the caller loads Acquire. Those RMWs form a
    /// release sequence, so the caller's Acquire read of the final `0`
    /// synchronizes-with **every** task's writes — all output-tile stores are
    /// visible after `run` returns. Spinning (vs a condvar) removes the
    /// per-barrier wakeup latency, which dominates when the tasks are the small
    /// GEMV tiles decode submits back-to-back.
    pub fn run(&self, tasks: Vec<Task>) {
        let count = tasks.len();
        if count <= 1 {
            // Nothing to distribute — run inline (no lock/atomic traffic).
            for t in tasks {
                t();
            }
            return;
        }
        // One batch owns the pool at a time (see `run_lock`). Uncontended
        // cost is one mutex acquisition per pooled kernel — nanoseconds
        // against the tasks it fences; contended, concurrent hosts serialize
        // their *parallel sections* (correct, and no slower than the
        // width-bounded pool could go anyway).
        let _batch = match self.run_lock.lock() {
            Ok(g) => g,
            // A poisoned lock only means an earlier batch panicked; its
            // `BatchDrain` already left the queue empty and its tasks
            // finished, so the pool state is clean and the lock is safe.
            Err(poisoned) => poisoned.into_inner(),
        };
        let remaining = Arc::new(AtomicUsize::new(count));
        let panicked = Arc::new(AtomicBool::new(false));
        // From here on, no batch task may outlive this call — even on an
        // unwind — the guard cancels everything still queued and awaits
        // everything in flight before this frame (and the operand pointers
        // the tasks capture) goes away.
        let drain = BatchDrain {
            shared: &self.shared,
            remaining: &remaining,
        };
        {
            let mut q = self.shared.queue.lock().unwrap();
            for t in tasks {
                let rem = Arc::clone(&remaining);
                let pan = Arc::clone(&panicked);
                q.push_back(Box::new(move || {
                    // Guard first: the decrement runs on normal return *and* on
                    // an unwinding `t()`, so the barrier can never deadlock.
                    let _done = Countdown(&rem);
                    // Contain a panicking task: a pool worker must survive it
                    // (the pool would otherwise shrink for the process
                    // lifetime) and a helping caller must not unwind mid-
                    // drain. The panic is re-raised on the publisher after
                    // the barrier — loud, on the walk that owns the task.
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(t)).is_err() {
                        pan.store(true, Ordering::Release);
                    }
                }));
            }
            // Publish the queue length under the lock, before any worker that
            // observes it can race to pop.
            self.shared.pending.fetch_add(count, Ordering::Release);
        }
        self.shared.cv.notify_all();
        // Calling thread helps drain. Pop in a scoped block so the queue
        // `MutexGuard` is released **before** the task runs — a `while let
        // Some(t) = lock().pop_front()` holds the guard across `t()` (the
        // condition temporary lives to the end of the loop body), which
        // serializes every worker behind the caller and silently turns the
        // pool into a sequential drain.
        loop {
            let next = {
                let mut q = self.shared.queue.lock().unwrap();
                let t = q.pop_front();
                if t.is_some() {
                    self.shared.pending.fetch_sub(1, Ordering::Relaxed);
                }
                t
            };
            match next {
                Some(task) => task(),
                None => break,
            }
        }
        // Barrier: the guard cancels anything still queued (none on this
        // path — the help-drain above emptied the queue) and waits for the
        // in-flight remainder. Workers finishing the last tiles are ~µs
        // away; the guard spins (PAUSE) briefly, then yields.
        drop(drain);
        if panicked.load(Ordering::Acquire) {
            panic!(
                "a pooled task panicked; its batch was cancelled and awaited                  (pool workers survive, the queue is clean)"
            );
        }
    }
}

fn worker_loop(shared: &Shared) {
    let mut idle: u32 = 0;
    loop {
        // Grab a task only when the lock-free hint says the queue is non-empty
        // (a pop that races another popper simply yields `None` and re-polls).
        let task = if shared.pending.load(Ordering::Acquire) > 0 {
            let mut q = shared.queue.lock().unwrap();
            let t = q.pop_front();
            if t.is_some() {
                shared.pending.fetch_sub(1, Ordering::Relaxed);
            }
            t
        } else {
            None
        };
        if let Some(t) = task {
            t();
            idle = 0;
            continue;
        }
        // No work: stay hot for the next back-to-back barrier, then park.
        if idle < WORKER_SPIN {
            idle += 1;
            std::hint::spin_loop();
            continue;
        }
        // Park until a submitter notifies; re-check the queue predicate under
        // the lock (the push happens under the lock before `notify_all`, so no
        // wakeup is missed).
        let mut q = shared.queue.lock().unwrap();
        while q.is_empty() {
            q = shared.cv.wait(q).unwrap();
        }
        let t = q.pop_front();
        if t.is_some() {
            shared.pending.fetch_sub(1, Ordering::Relaxed);
        }
        drop(q);
        if let Some(t) = t {
            t();
        }
        idle = 0;
    }
}

/// Bisect a `rows × cols` output into ≈`grain` disjoint tiles by repeatedly
/// halving the larger extent — the lattice recursion's frontier at the
/// parallel grain. Each tile is `(row0, rows, col0, cols)`. `col_align` keeps
/// column cuts on a panel boundary (the packed-weight layout requires it; pass
/// 1 for row-major). Tiles partition the output exactly and never overlap, so
/// concurrent writers are disjoint.
#[must_use]
pub fn output_tiles(
    rows: usize,
    cols: usize,
    grain: usize,
    col_align: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let mut tiles = vec![(0usize, rows, 0usize, cols)];
    while tiles.len() < grain {
        // Split the tile with the largest area along its longer splittable axis.
        let Some((idx, _)) = tiles
            .iter()
            .enumerate()
            .filter(|(_, &(_, r, _, c))| r > 1 || c >= 2 * col_align.max(1))
            .max_by_key(|(_, &(_, r, _, c))| r * c)
        else {
            break;
        };
        let (r0, r, c0, c) = tiles[idx];
        // Prefer splitting rows (no alignment constraint); else split columns
        // on a `col_align` boundary.
        if r >= 2 && r >= c / col_align.max(1) {
            let h = r / 2;
            tiles[idx] = (r0, h, c0, c);
            tiles.push((r0 + h, r - h, c0, c));
        } else if c >= 2 * col_align.max(1) {
            let half = (c / 2) / col_align.max(1) * col_align.max(1);
            let half = half.max(col_align.max(1));
            tiles[idx] = (r0, r, c0, half);
            tiles.push((r0, r, c0 + half, c - half));
        } else {
            break;
        }
    }
    tiles
}

/// `*mut T` / `*const T` that is `Send` — used to hand disjoint output regions
/// and shared read-only operands to pool tasks. **Safety contract:** the
/// pointed-to memory outlives the [`Pool::run`] barrier (the caller's buffers
/// are live for the whole synchronous kernel call), and the tiles handed out
/// by [`output_tiles`] are disjoint, so concurrent writers never alias.
#[derive(Clone, Copy)]
pub struct SendMut<T>(pub *mut T);
unsafe impl<T> Send for SendMut<T> {}

#[derive(Clone, Copy)]
pub struct SendConst<T>(pub *const T);
unsafe impl<T> Send for SendConst<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles must **exactly partition** the output — every cell covered once
    /// (the race-freedom witness for concurrent writers) — and honor the
    /// column alignment, across a spread of shapes / grains / alignments.
    #[test]
    fn output_tiles_partition_exactly_and_align() {
        for &(rows, cols) in &[
            (64usize, 1024usize),
            (1, 4096),
            (4096, 1),
            (200, 176),
            (7, 19),
            (64, 64),
        ] {
            for &grain in &[1usize, 2, 3, 4, 8, 16] {
                for &align in &[1usize, 16] {
                    let tiles = output_tiles(rows, cols, grain, align);
                    assert!(!tiles.is_empty());
                    let mut seen = vec![0u8; rows * cols];
                    for &(r0, r, c0, c) in &tiles {
                        assert!(r0 + r <= rows && c0 + c <= cols, "tile out of bounds");
                        assert!(r > 0 && c > 0, "empty tile");
                        if align > 1 {
                            assert_eq!(c0 % align, 0, "column offset not panel-aligned");
                            // Interior tiles (not the matrix's last column) keep
                            // whole panels; the final partial panel may be ragged.
                            if c0 + c != cols {
                                assert_eq!(c % align, 0, "interior column tile not aligned");
                            }
                        }
                        for i in r0..r0 + r {
                            for j in c0..c0 + c {
                                seen[i * cols + j] += 1;
                            }
                        }
                    }
                    assert!(
                        seen.iter().all(|&v| v == 1),
                        "{rows}×{cols} grain={grain} align={align}: not an exact partition"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod pool_diag {
    use super::*;
    use std::time::{Duration, Instant};

    /// The pool must run tasks **concurrently**, not drain them serially on
    /// the calling thread. Regression guard for the `while let Some(t) =
    /// lock().pop_front()` footgun (the queue guard outliving `t()` serialized
    /// every worker behind the caller). `width` independent ~80 ms spins must
    /// finish in well under the serial sum; we assert < 60% of serial, which a
    /// fully-serial drain (== serial) fails decisively while staying immune to
    /// scheduler jitter. Single-core hosts (width 1) skip — nothing to overlap.
    #[test]
    fn run_distributes_across_workers_concurrently() {
        let p = pool();
        let w = p.width();
        if w < 2 {
            return;
        }
        let spin = || {
            let t = Instant::now();
            while t.elapsed() < Duration::from_millis(80) {
                std::hint::spin_loop();
            }
        };
        let t = Instant::now();
        let tasks: Vec<Box<dyn FnOnce() + Send>> = (0..w)
            .map(|_| Box::new(spin) as Box<dyn FnOnce() + Send>)
            .collect();
        p.run(tasks);
        let wall = t.elapsed();
        let serial = Duration::from_millis(80 * w as u64);
        assert!(
            wall < serial.mul_f64(0.6),
            "pool ran {w} tasks in {wall:?}; serial≈{serial:?} — not concurrent \
             (the calling thread is holding the queue lock across tasks)"
        );
    }

    /// Barrier correctness (not timing): after `run` returns, **every** task
    /// has run **exactly once** and **every** disjoint write is visible to the
    /// caller. A dropped/duplicated task fails the run-count; a missing
    /// happens-before (the atomic release-sequence barrier) fails the readback.
    /// Swept over task counts that hit the inline path (0/1), exactly-`width`,
    /// `width`-relative fan-out, and a large fan-out, across many rounds to
    /// stress the spin/park/notify interleavings the rewrite introduced.
    #[test]
    fn run_executes_every_task_once_and_publishes_writes() {
        let p = pool();
        let w = p.width().max(1);
        let counts = [0usize, 1, 2, w, w + 1, 4 * w + 3, 97];
        for round in 0..300u32 {
            for &count in &counts {
                let mut out = vec![u32::MAX; count.max(1)];
                let ran = Arc::new(AtomicUsize::new(0));
                {
                    let op = SendMut(out.as_mut_ptr());
                    let tasks: Vec<Task> = (0..count)
                        .map(|i| {
                            let ran = Arc::clone(&ran);
                            Box::new(move || {
                                let op = op;
                                // Disjoint slot i; the value encodes (round, i)
                                // so a lost or duplicated task is detectable.
                                let v = round.wrapping_mul(1_000).wrapping_add(i as u32);
                                // SAFETY: each task owns a distinct slot i.
                                unsafe {
                                    *op.0.add(i) = v;
                                }
                                ran.fetch_add(1, Ordering::Relaxed);
                            }) as Task
                        })
                        .collect();
                    p.run(tasks);
                }
                assert_eq!(
                    ran.load(Ordering::Relaxed),
                    count,
                    "round {round} count {count}: task ran the wrong number of times"
                );
                for (i, &v) in out.iter().enumerate().take(count) {
                    assert_eq!(
                        v,
                        round.wrapping_mul(1_000).wrapping_add(i as u32),
                        "round {round} count {count}: slot {i} write not published after run()"
                    );
                }
            }
        }
    }

    /// The completion counter must reach zero even when a task unwinds — the
    /// `Countdown` guard's decrement-on-drop is what keeps the caller's barrier
    /// from spinning forever. Tested in isolation (a worker-thread panic in the
    /// process-wide pool would kill that worker and perturb other tests).
    #[test]
    fn countdown_decrements_on_normal_exit_and_unwind() {
        let c = AtomicUsize::new(2);
        {
            let _g = Countdown(&c);
        }
        assert_eq!(c.load(Ordering::Relaxed), 1, "guard must decrement on drop");
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = Countdown(&c);
            panic!("unwind past the guard");
        }));
        assert_eq!(
            c.load(Ordering::Relaxed),
            0,
            "guard must decrement even when the task unwinds"
        );
    }
}
