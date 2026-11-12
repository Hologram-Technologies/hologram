//! Parallel, multi-source, verified fetch of one large blob over HTTP-CAS.
//!
//! Given a κ and a list of untrusted sources, [`fetch_piece_tree`] takes the blob's 1 MiB piece
//! table from the first source whose table hashes to the κ, then [`fetch_pieces`] pulls every
//! missing piece with `Range` requests over keep-alive connections, many pieces at once, spread
//! across the sources. Each piece is checked against the table **before** it reaches the sink, so
//! a tool reading the sink never sees an unverified byte. When a source sends a piece that does not
//! verify, that source is dropped for the rest of the fetch and the piece is re-fetched from
//! another; a source that fails to answer is retried a few times, then dropped. Pieces the sink
//! already holds (a resumed download) are skipped.
//!
//! ```text
//! κ ──▶ fetch_piece_tree ──▶ PieceTree (checked against κ)
//!                              │
//!            ┌─────────────────┼─────────────────┐   N workers, Range: one piece each
//!         source A          source B          source C
//!            └──── verify_piece ──▶ PieceSink (e.g. PartialBlob) ──▶ commit
//! ```

extern crate std;

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use hologram_space::piece::piece_count;
use hologram_space::{KappaLabel71, PieceSink, PieceTree, StoreError, VerifyError};

use super::client::{ClientError, HttpConn};
use super::{cas_path, pieces_path};

/// Fetch settings.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Pieces in flight at once (one connection per worker per source).
    pub concurrency: usize,
    /// Transport failures (connect, timeout, bad status) a source may have before it is dropped.
    /// A source that sends one piece that does not verify is dropped at once.
    pub max_failures: u32,
    /// Socket read/write timeout.
    pub timeout: Duration,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            concurrency: 8,
            max_failures: 3,
            timeout: Duration::from_secs(30),
        }
    }
}

/// What one source contributed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceReport {
    pub source: String,
    /// Pieces this source delivered that verified.
    pub pieces: u64,
    pub bytes: u64,
    /// Transport failures.
    pub failures: u32,
    /// Pieces this source sent that did not verify (any is grounds to drop it).
    pub forged: u32,
    /// Dropped before the fetch finished.
    pub dropped: bool,
}

/// Outcome of a [`fetch_pieces`] run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FetchReport {
    /// Pieces fetched and verified in this run.
    pub pieces_fetched: u64,
    /// Pieces skipped because the sink already held them (resume).
    pub pieces_skipped: u64,
    pub bytes_fetched: u64,
    pub sources: Vec<SourceReport>,
}

/// Why a fetch failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchError {
    /// The κ is not a `blake3:` label (only BLAKE3 has a piece tree).
    NotBlake3,
    /// No source served a piece table that hashes to the κ.
    NoPieceTree,
    /// Every source still in play failed this piece (the rest of the download is kept).
    PieceUnavailable(u64),
    /// The sink refused a verified piece (e.g. disk full).
    Sink(StoreError),
    /// `done` does not have one entry per piece.
    BadResumeState,
}

/// Largest piece table accepted from a source: 24-byte header plus 32 bytes per MiB, so this
/// admits blobs up to 512 TiB.
const MAX_TABLE: u64 = 24 + 32 * (1 << 24);

/// Take the piece table for `kappa` from the first source whose table hashes to it. Sources whose
/// table does not match are skipped (and reported by the caller's next fetch, if they lie again).
pub fn fetch_piece_tree(
    kappa: &KappaLabel71,
    sources: &[String],
    opts: &FetchOptions,
) -> Result<PieceTree, FetchError> {
    hologram_space::piece::blake3_digest(kappa).map_err(|_| FetchError::NotBlake3)?;
    for s in sources {
        let mut conn = HttpConn::new(s, opts.timeout);
        let Ok(resp) = conn.request("GET", &pieces_path(kappa), &[], MAX_TABLE) else {
            continue;
        };
        if resp.status != 200 {
            continue;
        }
        if let Ok(tree) = PieceTree::decode(&resp.body) {
            if tree.verify_root(kappa).is_ok() {
                return Ok(tree);
            }
        }
    }
    Err(FetchError::NoPieceTree)
}

enum PieceError {
    /// The source answered with bytes that are not this piece.
    Forged,
    /// The source did not answer usefully (connect, timeout, status, protocol).
    Transport,
    /// The source does not hold the blob.
    Missing,
    /// The sink refused verified bytes.
    Sink(StoreError),
}

struct State {
    queue: VecDeque<u64>,
    /// Per queued piece: which sources already failed it.
    tried: Vec<Vec<bool>>,
    remaining: u64,
    in_flight: usize,
    cursor: usize,
    sources: Vec<SourceReport>,
    fatal: Option<FetchError>,
}

impl State {
    /// Next piece and the source to ask, or `None` when this worker should stop. Spreads requests
    /// round-robin over the sources still in play, never re-asking a source that failed the piece.
    fn next(&mut self) -> Option<Result<(u64, usize), ()>> {
        if self.fatal.is_some() || self.remaining == 0 {
            return None;
        }
        let piece = self.queue.pop_front()?;
        let n = self.sources.len();
        for step in 0..n {
            let s = (self.cursor + step) % n;
            if !self.sources[s].dropped && !self.tried[piece as usize][s] {
                self.cursor = (s + 1) % n;
                self.in_flight += 1;
                return Some(Ok((piece, s)));
            }
        }
        self.fatal = Some(FetchError::PieceUnavailable(piece));
        Some(Err(()))
    }
}

/// Fetch every piece of `tree` (which must already match `kappa`) that `done` does not mark, from
/// `sources`, verifying each and handing it to `sink`. `done` has one entry per piece; pass all
/// `false` for a fresh download, or the sink's own record when resuming.
pub fn fetch_pieces(
    kappa: &KappaLabel71,
    tree: &PieceTree,
    sources: &[String],
    done: &[bool],
    sink: &dyn PieceSink,
    opts: &FetchOptions,
) -> Result<FetchReport, FetchError> {
    tree.verify_root(kappa).map_err(|e| match e {
        VerifyError::NotBlake3 => FetchError::NotBlake3,
        _ => FetchError::NoPieceTree,
    })?;
    let n = tree.piece_count();
    if done.len() as u64 != n {
        return Err(FetchError::BadResumeState);
    }
    let todo: VecDeque<u64> = (0..n).filter(|&i| !done[i as usize]).collect();
    let skipped = n - todo.len() as u64;
    if let (true, Some(&first)) = (sources.is_empty(), todo.front()) {
        return Err(FetchError::PieceUnavailable(first));
    }
    let state = Mutex::new(State {
        remaining: todo.len() as u64,
        tried: (0..n)
            .map(|i| {
                if done[i as usize] {
                    Vec::new()
                } else {
                    alloc::vec![false; sources.len()]
                }
            })
            .collect(),
        queue: todo,
        in_flight: 0,
        cursor: 0,
        sources: sources
            .iter()
            .map(|s| SourceReport {
                source: s.clone(),
                ..SourceReport::default()
            })
            .collect(),
        fatal: None,
    });
    let wake = Condvar::new();
    let path = cas_path(kappa);
    let workers = opts.concurrency.max(1).min(n as usize);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut conns: Vec<HttpConn> = sources
                    .iter()
                    .map(|s| HttpConn::new(s, opts.timeout))
                    .collect();
                loop {
                    let job = {
                        let mut st = state.lock().unwrap();
                        loop {
                            if st.fatal.is_some() || st.remaining == 0 {
                                break None;
                            }
                            match st.next() {
                                Some(Ok(job)) => break Some(job),
                                Some(Err(())) => {
                                    wake.notify_all();
                                    break None;
                                }
                                // Queue empty but pieces in flight: one may come back for retry.
                                None => st = wake.wait(st).unwrap(),
                            }
                        }
                    };
                    let Some((piece, src)) = job else { break };
                    let result = fetch_one(&mut conns[src], &path, tree, piece, sink);
                    let mut st = state.lock().unwrap();
                    st.in_flight -= 1;
                    let len = tree.piece_range(piece).map_or(0, |r| r.end - r.start);
                    match result {
                        Ok(()) => {
                            st.remaining -= 1;
                            st.sources[src].pieces += 1;
                            st.sources[src].bytes += len;
                        }
                        Err(e) => {
                            match e {
                                PieceError::Forged => {
                                    st.sources[src].forged += 1;
                                    st.sources[src].dropped = true;
                                }
                                PieceError::Missing => {
                                    st.sources[src].failures += 1;
                                    st.sources[src].dropped = true;
                                }
                                PieceError::Transport => {
                                    st.sources[src].failures += 1;
                                    if st.sources[src].failures >= opts.max_failures {
                                        st.sources[src].dropped = true;
                                    }
                                }
                                PieceError::Sink(e) => st.fatal = Some(FetchError::Sink(e)),
                            }
                            st.tried[piece as usize][src] = true;
                            st.queue.push_front(piece);
                        }
                    }
                    drop(st);
                    wake.notify_all();
                }
            });
        }
    });

    let st = state.into_inner().unwrap();
    if let Some(e) = st.fatal {
        return Err(e);
    }
    Ok(FetchReport {
        pieces_fetched: st.sources.iter().map(|s| s.pieces).sum(),
        pieces_skipped: skipped,
        bytes_fetched: st.sources.iter().map(|s| s.bytes).sum(),
        sources: st.sources,
    })
}

fn fetch_one(
    conn: &mut HttpConn,
    path: &str,
    tree: &PieceTree,
    piece: u64,
    sink: &dyn PieceSink,
) -> Result<(), PieceError> {
    let r = tree.piece_range(piece).map_err(|_| PieceError::Transport)?;
    let want = r.end - r.start;
    let range = if want == 0 {
        // The empty blob: its one piece is empty; ask for the whole (zero-length) body.
        None
    } else {
        Some(std::format!("bytes={}-{}", r.start, r.end - 1))
    };
    let headers: Vec<(&str, &str)> = range.iter().map(|v| ("Range", v.as_str())).collect();
    // A body longer than the piece (e.g. a server that ignores `Range` and sends everything) is
    // refused before it is read: unusable, but not proof of a lie.
    let resp = conn
        .request("GET", path, &headers, want)
        .map_err(|_: ClientError| PieceError::Transport)?;
    match resp.status {
        206 => {}
        200 if r.start == 0 && r.end == tree.len() => {}
        // A 200 claims to be the whole blob, whose length the κ fixes: any other length is a lie.
        200 if resp.body.len() as u64 != tree.len() => return Err(PieceError::Forged),
        404 => return Err(PieceError::Missing),
        _ => return Err(PieceError::Transport),
    }
    if tree.verify_piece(piece, &resp.body).is_err() {
        return Err(PieceError::Forged);
    }
    sink.put_piece(piece, &resp.body).map_err(PieceError::Sink)
}

/// An in-memory [`PieceSink`] for blobs that fit in memory (and for tests).
pub struct MemPieceSink {
    buf: Mutex<Vec<u8>>,
    done: Mutex<Vec<bool>>,
}

impl MemPieceSink {
    /// A sink for a blob described by `tree`.
    pub fn new(tree: &PieceTree) -> Self {
        Self {
            buf: Mutex::new(alloc::vec![0u8; tree.len() as usize]),
            done: Mutex::new(alloc::vec![false; piece_count(tree.len()) as usize]),
        }
    }

    /// Which pieces have arrived.
    pub fn done(&self) -> Vec<bool> {
        self.done.lock().unwrap().clone()
    }

    /// The assembled bytes (meaningful once every piece is done).
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf.into_inner().unwrap()
    }
}

impl PieceSink for MemPieceSink {
    fn put_piece(&self, index: u64, bytes: &[u8]) -> Result<(), StoreError> {
        let start = (index * hologram_space::PIECE_LEN) as usize;
        let mut buf = self.buf.lock().unwrap();
        let end = start + bytes.len();
        if end > buf.len() {
            return Err(StoreError::OutOfRange);
        }
        buf[start..end].copy_from_slice(bytes);
        self.done.lock().unwrap()[index as usize] = true;
        Ok(())
    }
}

/// Convenience: fetch the whole blob into memory, verified. For blobs that fit in memory.
pub fn fetch_to_vec(
    kappa: &KappaLabel71,
    sources: &[String],
    opts: &FetchOptions,
) -> Result<(Vec<u8>, FetchReport), FetchError> {
    let tree = fetch_piece_tree(kappa, sources, opts)?;
    let sink = MemPieceSink::new(&tree);
    let done = alloc::vec![false; tree.piece_count() as usize];
    let report = fetch_pieces(kappa, &tree, sources, &done, &sink, opts)?;
    Ok((sink.into_bytes(), report))
}

impl core::fmt::Display for FetchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FetchError::NotBlake3 => f.write_str("κ is not a blake3 label"),
            FetchError::NoPieceTree => f.write_str("no source served a piece table matching the κ"),
            FetchError::PieceUnavailable(i) => write!(f, "no source delivered piece {i}"),
            FetchError::Sink(e) => write!(f, "sink refused a piece: {e:?}"),
            FetchError::BadResumeState => f.write_str("resume state has the wrong piece count"),
        }
    }
}

impl std::error::Error for FetchError {}
