//! # hologram_store::file — one file per κ, for multi-gigabyte blobs
//!
//! A [`KappaStore`] over a plain directory, built for blobs too large to hold in memory (model
//! weights). Ported from `hologram-ai`'s `DirKappaStore` and extended:
//!
//! - **Atomic writes.** Every blob is written to `tmp/`, flushed to disk, then renamed into
//!   `blobs/`. A crash leaves either the whole blob or nothing under its κ; `open` clears `tmp/`.
//! - **Verified streaming put.** [`KappaStore::put_writer`] hashes bytes as they are written and,
//!   given an expected κ, refuses to install bytes that do not match it.
//! - **Ranged reads** ([`KappaStore::get_range`], [`KappaStore::content_len`]) seek into the file;
//!   nothing else is read.
//! - **Piece tables.** [`FileKappaStore::piece_tree`] returns the blob's 1 MiB [`PieceTree`]
//!   (computed once, in parallel, and cached under `trees/`), which a server hands to clients so
//!   they can verify ranges.
//! - **Resumable out-of-order downloads.** [`FileKappaStore::begin_partial`] opens a
//!   [`PartialBlob`] that accepts verified pieces in any order from any number of threads, survives
//!   a restart (pieces already on disk are re-verified, never trusted), and installs the blob on
//!   [`PartialBlob::commit`].
//! - **Pins and reachability GC**, with the same semantics as the in-memory reference.
//!
//! Layout under the root (κ `blake3:<hex>` is spelled `blake3-<hex>`, because `:` names an
//! alternate data stream on NTFS):
//!
//! ```text
//! blobs/blake3-<hex>          content
//! trees/blake3-<hex>          piece table (blobs over 1 MiB), checked against the κ when read
//! pins/blake3-<hex>           empty marker: a reachability root
//! partial/blake3-<hex>.part   an unfinished download, its piece table in .tree
//! tmp/                        in-flight writes
//! ```
//!
//! Only the BLAKE3 σ-axis is stored (the same scope as the redb backend); other axes fail with
//! [`StoreError::UnknownAxis`].

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use hologram_space::piece::{piece_chaining_value, piece_count};
use hologram_space::{
    references, Bytes, GarbageCollect, KappaLabel, KappaLabel71, KappaStore, KappaWriter,
    PieceSink, PieceTree, PieceTreeBuilder, RealizationRegistry, StoreError, StreamVerifier,
    PIECE_LEN,
};

fn io_err(e: std::io::Error) -> StoreError {
    match e.kind() {
        ErrorKind::StorageFull | ErrorKind::QuotaExceeded => StoreError::QuotaExceeded,
        _ => StoreError::BackendFailure("filesystem"),
    }
}

/// File name of a κ: `blake3:<hex>` → `blake3-<hex>`.
pub fn kappa_file_name(kappa: &KappaLabel71) -> String {
    kappa.as_str().replacen(':', "-", 1)
}

/// Inverse of [`kappa_file_name`]; `None` for anything that is not a κ file name.
pub fn parse_kappa_file_name(name: &str) -> Option<KappaLabel71> {
    let label = name.replacen('-', ":", 1);
    let bytes: [u8; 71] = label.as_bytes().try_into().ok()?;
    KappaLabel::from_bytes(&bytes).ok()
}

/// A content-addressed store of one file per κ under a root directory. Cheap to share: every
/// method takes `&self` and the type is `Send + Sync`.
pub struct FileKappaStore {
    root: PathBuf,
    seq: AtomicU64,
    cache: std::sync::Mutex<SmallCache>,
    handles: std::sync::Mutex<HandleCache>,
}

/// Blobs up to this size are kept in the read cache, so repeated `get`s of small content (manifests,
/// configs) share one `Arc` (the TCK's zero-copy floor). Larger blobs are read with `get_range`.
pub const CACHE_ENTRY_MAX: u64 = 1 << 20;

/// Total bytes the read cache may hold.
pub const CACHE_TOTAL_MAX: u64 = 64 << 20;

/// First-in-first-out cache of small blobs, bounded by [`CACHE_TOTAL_MAX`].
#[derive(Default)]
struct SmallCache {
    map: std::collections::HashMap<[u8; 71], Bytes>,
    order: std::collections::VecDeque<[u8; 71]>,
    bytes: u64,
}

impl SmallCache {
    fn get(&self, k: &[u8; 71]) -> Option<Bytes> {
        self.map.get(k).cloned()
    }

    fn insert(&mut self, k: [u8; 71], v: Bytes) {
        if v.len() as u64 > CACHE_ENTRY_MAX || self.map.contains_key(&k) {
            return;
        }
        self.bytes += v.len() as u64;
        self.map.insert(k, v);
        self.order.push_back(k);
        while self.bytes > CACHE_TOTAL_MAX {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(b) = self.map.remove(&old) {
                self.bytes -= b.len() as u64;
            }
        }
    }

    fn remove(&mut self, k: &[u8; 71]) {
        if let Some(b) = self.map.remove(k) {
            self.bytes -= b.len() as u64;
            self.order.retain(|x| x != k);
        }
    }
}

impl FileKappaStore {
    /// Open (creating if needed) a store rooted at `root`. Leftover temporary files from an
    /// interrupted process are removed; unfinished downloads under `partial/` are kept.
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        for d in ["blobs", "trees", "pins", "partial", "tmp"] {
            fs::create_dir_all(root.join(d))?;
        }
        for entry in fs::read_dir(root.join("tmp"))? {
            let _ = fs::remove_file(entry?.path());
        }
        Ok(Self {
            root,
            seq: AtomicU64::new(0),
            cache: std::sync::Mutex::new(SmallCache::default()),
            handles: std::sync::Mutex::new(HandleCache::default()),
        })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of the file holding `kappa`'s content (it may not exist).
    pub fn blob_path(&self, kappa: &KappaLabel71) -> PathBuf {
        self.root.join("blobs").join(kappa_file_name(kappa))
    }

    fn tree_path(&self, kappa: &KappaLabel71) -> PathBuf {
        self.root.join("trees").join(kappa_file_name(kappa))
    }

    fn pin_path(&self, kappa: &KappaLabel71) -> PathBuf {
        self.root.join("pins").join(kappa_file_name(kappa))
    }

    fn partial_path(&self, kappa: &KappaLabel71, ext: &str) -> PathBuf {
        self.root
            .join("partial")
            .join(format!("{}.{ext}", kappa_file_name(kappa)))
    }

    fn tmp_path(&self) -> PathBuf {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        self.root
            .join("tmp")
            .join(format!("{}-{n}.tmp", std::process::id()))
    }

    /// A shared read handle on `kappa`'s file and its length, opened once and reused (content
    /// under a κ never changes, so neither does the length). `None` if absent.
    fn handle(&self, kappa: &KappaLabel71) -> Result<Option<(Arc<File>, u64)>, StoreError> {
        if let Some(h) = self.handles.lock().unwrap().get(kappa.as_array()) {
            return Ok(Some(h));
        }
        let f = match File::open(self.blob_path(kappa)) {
            Ok(f) => Arc::new(f),
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err(e)),
        };
        let len = f.metadata().map_err(io_err)?.len();
        self.handles
            .lock()
            .unwrap()
            .insert(*kappa.as_array(), (f.clone(), len));
        Ok(Some((f, len)))
    }

    /// Open `kappa`'s content for reading (e.g. to stream it to a socket). `None` if absent.
    pub fn open_blob(&self, kappa: &KappaLabel71) -> Result<Option<File>, StoreError> {
        match File::open(self.blob_path(kappa)) {
            Ok(f) => Ok(Some(f)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_err(e)),
        }
    }

    /// Move a finished temporary file into place as `dest`. If `dest` already exists the content
    /// is identical by construction (same κ), so the temporary file is dropped.
    fn install(tmp: &Path, dest: &Path) -> Result<(), StoreError> {
        if dest.exists() {
            let _ = fs::remove_file(tmp);
            return Ok(());
        }
        match fs::rename(tmp, dest) {
            Ok(()) => Ok(()),
            Err(_) if dest.exists() => {
                let _ = fs::remove_file(tmp);
                Ok(())
            }
            Err(e) => {
                let _ = fs::remove_file(tmp);
                Err(io_err(e))
            }
        }
    }

    /// Write `bytes` to a temporary file, flush it to disk, and rename it to `dest`.
    fn write_atomic(&self, dest: &Path, bytes: &[u8]) -> Result<(), StoreError> {
        let tmp = self.tmp_path();
        let mut f = File::create(&tmp).map_err(io_err)?;
        f.write_all(bytes).map_err(io_err)?;
        f.sync_all().map_err(io_err)?;
        drop(f);
        Self::install(&tmp, dest)
    }

    /// The blob's 1 MiB piece table: read from `trees/` if cached (and checked against the κ),
    /// otherwise computed from the file on all cores and cached. `None` if the blob is absent;
    /// [`StoreError::KappaMismatch`] if the bytes on disk no longer hash to their κ.
    pub fn piece_tree(&self, kappa: &KappaLabel71) -> Result<Option<PieceTree>, StoreError> {
        let Some(len) = self.content_len(kappa)? else {
            return Ok(None);
        };
        if let Ok(bytes) = fs::read(self.tree_path(kappa)) {
            if let Ok(tree) = PieceTree::decode(&bytes) {
                if tree.len() == len && tree.verify_root(kappa).is_ok() {
                    return Ok(Some(tree));
                }
            }
            let _ = fs::remove_file(self.tree_path(kappa));
        }
        let tree = tree_of_file(&self.blob_path(kappa), len)?;
        tree.verify_root(kappa)?;
        if tree.piece_count() > 1 {
            self.write_atomic(&self.tree_path(kappa), &tree.encode())?;
        }
        Ok(Some(tree))
    }

    /// Re-hash `kappa`'s file and confirm it still matches its κ (a scrub). `None` if absent.
    pub fn verify_blob(&self, kappa: &KappaLabel71) -> Result<Option<bool>, StoreError> {
        let Some(mut f) = self.open_blob(kappa)? else {
            return Ok(None);
        };
        let mut v = StreamVerifier::new(kappa)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf).map_err(io_err)?;
            if n == 0 {
                break;
            }
            v.update(&buf[..n])?;
        }
        Ok(Some(v.finish().is_ok()))
    }

    /// Start (or resume) an out-of-order download of `kappa`, whose piece table is `tree`. The
    /// table must hash to `kappa` ([`StoreError::KappaMismatch`] otherwise). If a download of this
    /// κ was interrupted, the pieces already on disk are re-verified and kept.
    pub fn begin_partial(
        &self,
        kappa: &KappaLabel71,
        tree: PieceTree,
    ) -> Result<PartialBlob<'_>, StoreError> {
        tree.verify_root(kappa)?;
        let n = tree.piece_count() as usize;
        let done: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        let part = self.partial_path(kappa, "part");
        if self.contains(kappa) {
            done.iter().for_each(|d| d.store(true, Ordering::Relaxed));
            return Ok(PartialBlob {
                store: self,
                kappa: *kappa,
                tree,
                part,
                file: None,
                done,
                installed: true,
            });
        }
        // The file is not preallocated: pieces arrive nearly in order, so each write extends it a
        // little. (On NTFS a write past the valid-data length first zero-fills the gap, so a
        // preallocated file made every out-of-order write pay for the bytes before it.)
        match fs::metadata(&part).map(|m| m.len()) {
            Ok(have) if have <= tree.len() => verify_pieces_on_disk(&part, &tree, have, &done)?,
            _ => drop(File::create(&part).map_err(io_err)?),
        }
        self.write_atomic(&self.partial_path(kappa, "tree"), &tree.encode())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&part)
            .map_err(io_err)?;
        Ok(PartialBlob {
            store: self,
            kappa: *kappa,
            tree,
            part,
            file: Some(file),
            done,
            installed: false,
        })
    }

    /// Resume an interrupted download using the piece table saved beside it. `None` if there is no
    /// unfinished download of `kappa`.
    pub fn resume_partial(
        &self,
        kappa: &KappaLabel71,
    ) -> Result<Option<PartialBlob<'_>>, StoreError> {
        match fs::read(self.partial_path(kappa, "tree")) {
            Ok(bytes) => {
                let tree = PieceTree::decode(&bytes)?;
                self.begin_partial(kappa, tree).map(Some)
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_err(e)),
        }
    }

    /// The κs with an unfinished download under `partial/`.
    pub fn partials(&self) -> Vec<KappaLabel71> {
        list_kappas(&self.root.join("partial"), ".tree")
    }

    /// The reachable closure from the pinned roots (the GC mark set). A blob contributes edges only
    /// if it begins with a registered realization IRI, so large opaque blobs are never read whole.
    pub fn reachable(
        &self,
        registry: RealizationRegistry<'_>,
    ) -> Result<std::collections::HashSet<[u8; 71]>, StoreError> {
        let mut live = std::collections::HashSet::new();
        let mut frontier: Vec<KappaLabel71> = self.pinned_roots();
        while let Some(k) = frontier.pop() {
            if !live.insert(*k.as_array()) {
                continue;
            }
            if !self.starts_with_known_iri(&k, registry)? {
                continue;
            }
            if let Some(bytes) = self.get(&k)? {
                if let Ok(refs) = references(&bytes, registry) {
                    frontier.extend(refs);
                }
            }
        }
        Ok(live)
    }

    fn starts_with_known_iri(
        &self,
        kappa: &KappaLabel71,
        registry: RealizationRegistry<'_>,
    ) -> Result<bool, StoreError> {
        let Some(len) = self.content_len(kappa)? else {
            return Ok(false);
        };
        let longest = registry.iter().map(|(iri, _)| iri.len()).max().unwrap_or(0);
        let head_len = len.min(longest as u64 + 1);
        let Some(head) = self.get_range(kappa, 0, head_len)? else {
            return Ok(false);
        };
        Ok(registry.iter().any(|(iri, _)| {
            head.len() > iri.len() && head.starts_with(iri.as_bytes()) && head[iri.len()] == 0
        }))
    }

    /// Delete every blob not reachable from a pin (and its cached piece table). Returns the count.
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        let live = self.reachable(registry)?;
        let mut evicted = 0;
        for k in self.iterate() {
            if !live.contains(k.as_array()) {
                self.cache.lock().unwrap().remove(k.as_array());
                self.handles.lock().unwrap().remove(k.as_array());
                match fs::remove_file(self.blob_path(&k)) {
                    Ok(()) => evicted += 1,
                    Err(e) if e.kind() == ErrorKind::NotFound => {}
                    Err(e) => return Err(io_err(e)),
                }
                let _ = fs::remove_file(self.tree_path(&k));
            }
        }
        Ok(evicted)
    }
}

fn list_kappas(dir: &Path, suffix: &str) -> Vec<KappaLabel71> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            parse_kappa_file_name(name.strip_suffix(suffix)?)
        })
        .collect()
}

// Positional I/O on a shared handle: no seek, so threads can share one `File`. Opening a file is
// far from free (on Windows it can cost milliseconds when an on-access scanner inspects a large
// file), so the store opens each blob once and reads it at offsets.

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, offset)
}

#[cfg(unix)]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(f, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, offset) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(f: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_write(buf, offset) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Other targets (e.g. WASI): seek and read under one lock, so offsets never interleave.
#[cfg(not(any(unix, windows)))]
static POSITIONAL_IO: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(not(any(unix, windows)))]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let _g = POSITIONAL_IO.lock().unwrap();
    let mut f = f;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

#[cfg(not(any(unix, windows)))]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let _g = POSITIONAL_IO.lock().unwrap();
    let mut f = f;
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(buf)
}

/// Run `work(file, piece)` for every piece index on all cores, each thread with its own handle on
/// `path` (one open per thread, not per piece); first error wins.
fn for_each_piece_parallel(
    path: &Path,
    pieces: u64,
    work: impl Fn(&File, u64) -> Result<(), StoreError> + Sync,
) -> Result<(), StoreError> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .min(pieces.max(1) as usize);
    let next = AtomicU64::new(0);
    let failed = std::sync::Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let file = match File::open(path) {
                    Ok(f) => f,
                    Err(e) => {
                        *failed.lock().unwrap() = Some(io_err(e));
                        return;
                    }
                };
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pieces || failed.lock().unwrap().is_some() {
                        break;
                    }
                    if let Err(e) = work(&file, i) {
                        *failed.lock().unwrap() = Some(e);
                        break;
                    }
                }
            });
        }
    });
    match failed.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn read_piece(file: &File, tree_len: u64, i: u64) -> Result<Vec<u8>, StoreError> {
    let start = i * PIECE_LEN;
    let mut buf = vec![0u8; (tree_len - start).min(PIECE_LEN) as usize];
    read_exact_at(file, &mut buf, start).map_err(io_err)?;
    Ok(buf)
}

/// Compute the piece table of the `len`-byte file at `path`, one piece per task on all cores.
fn tree_of_file(path: &Path, len: u64) -> Result<PieceTree, StoreError> {
    let n = piece_count(len);
    let cvs: Vec<std::sync::Mutex<[u8; 32]>> =
        (0..n).map(|_| std::sync::Mutex::new([0u8; 32])).collect();
    for_each_piece_parallel(path, n, |file, i| {
        let bytes = read_piece(file, len, i)?;
        *cvs[i as usize].lock().unwrap() = piece_chaining_value(i, &bytes, n);
        Ok(())
    })?;
    let cvs = cvs.into_iter().map(|m| m.into_inner().unwrap()).collect();
    Ok(PieceTree::from_parts(len, cvs)?)
}

/// Mark `done[i]` for every piece of `part` (whose first `have` bytes exist) that already verifies
/// against `tree`. A piece not wholly on disk is missing.
fn verify_pieces_on_disk(
    part: &Path,
    tree: &PieceTree,
    have: u64,
    done: &[AtomicBool],
) -> Result<(), StoreError> {
    for_each_piece_parallel(part, tree.piece_count(), |file, i| {
        if tree.piece_range(i)?.end > have || (have == 0 && !tree.is_empty()) {
            return Ok(());
        }
        let bytes = read_piece(file, tree.len(), i)?;
        if tree.verify_piece(i, &bytes).is_ok() {
            done[i as usize].store(true, Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Open read handles by κ, bounded; the oldest is closed when the cap is reached.
#[derive(Default)]
struct HandleCache {
    map: std::collections::HashMap<[u8; 71], (Arc<File>, u64)>,
    order: std::collections::VecDeque<[u8; 71]>,
}

/// Read handles kept open at once.
const OPEN_HANDLES_MAX: usize = 64;

impl HandleCache {
    fn get(&self, k: &[u8; 71]) -> Option<(Arc<File>, u64)> {
        self.map.get(k).cloned()
    }

    fn insert(&mut self, k: [u8; 71], f: (Arc<File>, u64)) {
        if self.map.insert(k, f).is_none() {
            self.order.push_back(k);
        }
        while self.order.len() > OPEN_HANDLES_MAX {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }

    fn remove(&mut self, k: &[u8; 71]) {
        if self.map.remove(k).is_some() {
            self.order.retain(|x| x != k);
        }
    }
}

impl KappaStore for FileKappaStore {
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError> {
        let mut w = self.put_writer(axis, None)?;
        w.write(canonical_bytes)?;
        w.finish()
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        if let Some(b) = self.cache.lock().unwrap().get(kappa.as_array()) {
            return Ok(Some(b));
        }
        match fs::read(self.blob_path(kappa)) {
            Ok(v) => {
                let b: Bytes = Arc::from(v);
                self.cache
                    .lock()
                    .unwrap()
                    .insert(*kappa.as_array(), b.clone());
                Ok(Some(b))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_err(e)),
        }
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        self.blob_path(kappa).is_file()
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        File::create(self.pin_path(kappa)).map_err(io_err)?;
        Ok(())
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        match fs::remove_file(self.pin_path(kappa)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => Err(StoreError::NotPinned),
            Err(e) => Err(io_err(e)),
        }
    }

    fn iterate(&self) -> Vec<KappaLabel71> {
        list_kappas(&self.root.join("blobs"), "")
    }

    fn pinned_roots(&self) -> Vec<KappaLabel71> {
        list_kappas(&self.root.join("pins"), "")
    }

    fn approximate_count(&self) -> usize {
        self.iterate().len()
    }

    fn approximate_bytes(&self) -> u64 {
        fs::read_dir(self.root.join("blobs"))
            .map(|rd| {
                rd.filter_map(|e| e.ok()?.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    fn content_len(&self, kappa: &KappaLabel71) -> Result<Option<u64>, StoreError> {
        if let Some((_, len)) = self.handles.lock().unwrap().get(kappa.as_array()) {
            return Ok(Some(len));
        }
        match fs::metadata(self.blob_path(kappa)) {
            Ok(m) => Ok(Some(m.len())),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_err(e)),
        }
    }

    fn get_range(
        &self,
        kappa: &KappaLabel71,
        offset: u64,
        len: u64,
    ) -> Result<Option<Bytes>, StoreError> {
        let Some((f, total)) = self.handle(kappa)? else {
            return Ok(None);
        };
        let end = offset.checked_add(len).ok_or(StoreError::OutOfRange)?;
        if end > total {
            return Err(StoreError::OutOfRange);
        }
        // Read straight into the shared buffer the caller gets: one allocation, no copy.
        let mut out: Bytes = std::iter::repeat_n(0u8, len as usize).collect();
        let buf = Arc::get_mut(&mut out).expect("freshly allocated, so unique");
        read_exact_at(&f, buf, offset).map_err(io_err)?;
        Ok(Some(out))
    }

    fn put_writer<'a>(
        &'a self,
        axis: &str,
        expected: Option<&KappaLabel71>,
    ) -> Result<Box<dyn KappaWriter + 'a>, StoreError> {
        if axis != "blake3" {
            return Err(StoreError::UnknownAxis);
        }
        let tmp = self.tmp_path();
        let file = File::create(&tmp).map_err(io_err)?;
        Ok(Box::new(FileWriter {
            store: self,
            tmp,
            file: Some(std::io::BufWriter::with_capacity(1 << 20, file)),
            tree: PieceTreeBuilder::new(),
            expected: expected.copied(),
        }))
    }
}

impl GarbageCollect for FileKappaStore {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        FileKappaStore::gc(self, registry)
    }
}

/// A streaming put into a [`FileKappaStore`]: bytes go to a temporary file while their piece
/// table is built; `finish` checks the κ, flushes, and renames the file into place.
struct FileWriter<'a> {
    store: &'a FileKappaStore,
    tmp: PathBuf,
    file: Option<std::io::BufWriter<File>>,
    tree: PieceTreeBuilder,
    expected: Option<KappaLabel71>,
}

impl KappaWriter for FileWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        self.tree.update(bytes);
        self.file
            .as_mut()
            .expect("writer is live until finish")
            .write_all(bytes)
            .map_err(io_err)
    }

    fn finish(mut self: Box<Self>) -> Result<KappaLabel71, StoreError> {
        let tree = core::mem::take(&mut self.tree).finish();
        let kappa = tree.kappa();
        if let Some(want) = &self.expected {
            if *want != kappa {
                return Err(StoreError::KappaMismatch); // Drop removes the temporary file.
            }
        }
        let file = self
            .file
            .take()
            .expect("writer is live until finish")
            .into_inner()
            .map_err(|e| io_err(e.into_error()))?;
        file.sync_all().map_err(io_err)?;
        drop(file);
        FileKappaStore::install(&self.tmp, &self.store.blob_path(&kappa))?;
        if tree.piece_count() > 1 {
            let _ = self
                .store
                .write_atomic(&self.store.tree_path(&kappa), &tree.encode());
        }
        Ok(kappa)
    }
}

impl Drop for FileWriter<'_> {
    fn drop(&mut self) {
        // Either finish already renamed the file away, or the write was abandoned or refused.
        self.file.take();
        let _ = fs::remove_file(&self.tmp);
    }
}

/// An unfinished download: a preallocated file under `partial/` that accepts pieces in any order,
/// from any thread, each verified against the piece table before it touches the disk.
pub struct PartialBlob<'a> {
    store: &'a FileKappaStore,
    kappa: KappaLabel71,
    tree: PieceTree,
    part: PathBuf,
    /// Read-write handle on `part`, opened once (`None` once the blob is installed).
    file: Option<File>,
    done: Vec<AtomicBool>,
    installed: bool,
}

impl PartialBlob<'_> {
    /// The κ being downloaded.
    pub fn kappa(&self) -> &KappaLabel71 {
        &self.kappa
    }

    /// The piece table (already checked against the κ).
    pub fn tree(&self) -> &PieceTree {
        &self.tree
    }

    /// Which pieces are on disk and verified.
    pub fn done(&self) -> Vec<bool> {
        self.done
            .iter()
            .map(|d| d.load(Ordering::Acquire))
            .collect()
    }

    /// Indices of the pieces still missing.
    pub fn missing(&self) -> Vec<u64> {
        (0..self.done.len() as u64)
            .filter(|&i| !self.done[i as usize].load(Ordering::Acquire))
            .collect()
    }

    /// True once every piece is on disk and verified.
    pub fn is_complete(&self) -> bool {
        self.done.iter().all(|d| d.load(Ordering::Acquire))
    }

    /// Flush the file and install it under its κ. Fails with `BackendFailure` if pieces are
    /// missing (the download stays resumable).
    pub fn commit(mut self) -> Result<KappaLabel71, StoreError> {
        if self.installed {
            return Ok(self.kappa);
        }
        if !self.is_complete() {
            return Err(StoreError::BackendFailure("partial blob incomplete"));
        }
        let f = self
            .file
            .take()
            .expect("an uninstalled partial holds its file");
        f.sync_all().map_err(io_err)?;
        drop(f);
        FileKappaStore::install(&self.part, &self.store.blob_path(&self.kappa))?;
        let tree_tmp = self.store.partial_path(&self.kappa, "tree");
        if self.tree.piece_count() > 1 {
            FileKappaStore::install(&tree_tmp, &self.store.tree_path(&self.kappa))?;
        } else {
            let _ = fs::remove_file(tree_tmp);
        }
        Ok(self.kappa)
    }

    /// Give up the download and delete what was written.
    pub fn abort(mut self) {
        self.file.take();
        if !self.installed {
            let _ = fs::remove_file(&self.part);
            let _ = fs::remove_file(self.store.partial_path(&self.kappa, "tree"));
        }
    }
}

impl PieceSink for PartialBlob<'_> {
    fn put_piece(&self, index: u64, bytes: &[u8]) -> Result<(), StoreError> {
        // Verified here as well as by the caller: nothing reaches the disk unverified.
        self.tree.verify_piece(index, bytes)?;
        if self.installed || self.done[index as usize].load(Ordering::Acquire) {
            return Ok(());
        }
        let f = self
            .file
            .as_ref()
            .expect("an uninstalled partial holds its file");
        write_all_at(f, bytes, index * PIECE_LEN).map_err(io_err)?;
        self.done[index as usize].store(true, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_roundtrip_and_reject_junk() {
        let k = hologram_space::address_bytes(b"x");
        assert_eq!(parse_kappa_file_name(&kappa_file_name(&k)), Some(k));
        assert!(!kappa_file_name(&k).contains(':'));
        assert_eq!(parse_kappa_file_name("blake3-xyz"), None);
        assert_eq!(parse_kappa_file_name("README"), None);
    }
}
