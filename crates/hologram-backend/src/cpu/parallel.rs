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
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;

type Task = Box<dyn FnOnce() + Send>;

struct Shared {
    queue: Mutex<VecDeque<Task>>,
    cv: Condvar,
}

/// A persistent set of worker threads sharing one task queue.
pub struct Pool {
    shared: Arc<Shared>,
    width: usize,
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
        });
        // `width - 1` persistent workers; the submitting thread is the width-th
        // runner (it drains the queue in `run`), so a 1-core host spawns none.
        for _ in 1..width {
            let sh = Arc::clone(&shared);
            thread::spawn(move || worker_loop(&sh));
        }
        Pool { shared, width }
    }

    /// Number of concurrent runners (cores).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Run `tasks` to completion. The calling thread participates (drains the
    /// queue), so flat task sets never deadlock regardless of `width`. Returns
    /// only after every task has finished; the completion barrier (a mutex
    /// release/acquire) establishes happens-before, so all task writes are
    /// visible to the caller afterwards.
    pub fn run(&self, tasks: Vec<Task>) {
        if tasks.len() <= 1 {
            // Nothing to distribute — run inline (no lock traffic).
            for t in tasks {
                t();
            }
            return;
        }
        let remaining = Arc::new((Mutex::new(tasks.len()), Condvar::new()));
        {
            let mut q = self.shared.queue.lock().unwrap();
            for t in tasks {
                let rem = Arc::clone(&remaining);
                q.push_back(Box::new(move || {
                    t();
                    let (m, cv) = &*rem;
                    let mut g = m.lock().unwrap();
                    *g -= 1;
                    if *g == 0 {
                        cv.notify_all();
                    }
                }));
            }
        }
        self.shared.cv.notify_all();
        // Calling thread helps drain.
        while let Some(task) = self.shared.queue.lock().unwrap().pop_front() {
            task();
        }
        // Wait for tasks still running on workers.
        let (m, cv) = &*remaining;
        let mut g = m.lock().unwrap();
        while *g != 0 {
            g = cv.wait(g).unwrap();
        }
    }
}

fn worker_loop(shared: &Shared) {
    loop {
        let task = {
            let mut q = shared.queue.lock().unwrap();
            loop {
                if let Some(t) = q.pop_front() {
                    break t;
                }
                q = shared.cv.wait(q).unwrap();
            }
        };
        task();
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
