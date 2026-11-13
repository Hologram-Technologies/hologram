#![cfg(feature = "file")]
//! The file-per-κ backend: the shared TCK, reachability GC, durability, and the large-blob
//! behaviour it exists for (streaming verified put, ranged reads, piece tables, resumable
//! out-of-order downloads). Each test plants a defect the store must refuse.

use std::path::PathBuf;

use hologram_space::piece::piece_count;
use hologram_space::{
    address_bytes, ContainerManifest, KappaStore, PieceSink, PieceTree, Realization, StoreError,
    PIECE_LEN, REGISTRY,
};
use hologram_store::file::{kappa_file_name, FileKappaStore};
use hologram_tck::store_battery;

/// A fresh directory under the system temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "hologram-file-store-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn blob(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn tmp_is_empty(dir: &TempDir) -> bool {
    std::fs::read_dir(dir.0.join("tmp"))
        .unwrap()
        .next()
        .is_none()
}

#[test]
fn file_store_passes_the_kappastore_tck() {
    let dir = TempDir::new("tck");
    let store = FileKappaStore::open(&dir.0).unwrap();
    store_battery(&store);
    assert!(tmp_is_empty(&dir), "no temporary file outlives a write");
}

#[test]
fn file_reachability_gc_matches_reference_semantics() {
    let dir = TempDir::new("gc");
    let store = FileKappaStore::open(&dir.0).unwrap();
    let code = store.put("blake3", b"wasm").unwrap();
    let state = store.put("blake3", b"state").unwrap();
    let params = store.put("blake3", b"params").unwrap();
    let manifest = ContainerManifest {
        code,
        initial_state: state,
        parameters: params,
    };
    let mk = store.put("blake3", &manifest.canonicalize()).unwrap();
    let orphan = store.put("blake3", b"orphan").unwrap();
    let big = store
        .put("blake3", &blob(3 * PIECE_LEN as usize, 1))
        .unwrap();
    store.pin(&mk).unwrap();
    store.pin(&big).unwrap();

    assert_eq!(store.gc(REGISTRY).unwrap(), 1);
    for k in [mk, code, state, params, big] {
        assert!(store.contains(&k));
    }
    assert!(!store.contains(&orphan));
    assert_eq!(
        store.get(&orphan).unwrap(),
        None,
        "the read cache was invalidated"
    );

    store.unpin(&big).unwrap();
    assert_eq!(store.gc(REGISTRY).unwrap(), 1);
    assert!(!store.contains(&big));
}

#[test]
fn file_store_persists_and_clears_stale_temporaries_on_reopen() {
    let dir = TempDir::new("reopen");
    let k = {
        let store = FileKappaStore::open(&dir.0).unwrap();
        let k = store.put("blake3", b"durable").unwrap();
        store.pin(&k).unwrap();
        k
    };
    // Planted defect: a crash left a half-written temporary file behind.
    std::fs::write(dir.0.join("tmp").join("999-0.tmp"), b"half a blob").unwrap();
    let store = FileKappaStore::open(&dir.0).unwrap();
    assert_eq!(store.get(&k).unwrap().unwrap().as_ref(), b"durable");
    assert_eq!(store.pinned_roots(), vec![k]);
    assert!(tmp_is_empty(&dir));
}

#[test]
fn streaming_put_ranged_reads_and_piece_table_on_a_multi_piece_blob() {
    let dir = TempDir::new("large");
    let store = FileKappaStore::open(&dir.0).unwrap();
    let b = blob(5 * PIECE_LEN as usize + 333, 7);
    let k = address_bytes(&b);

    let mut w = store.put_writer("blake3", Some(&k)).unwrap();
    for part in b.chunks(300_007) {
        w.write(part).unwrap();
    }
    assert_eq!(w.finish().unwrap(), k);
    assert_eq!(store.content_len(&k).unwrap(), Some(b.len() as u64));

    for (off, len) in [
        (0u64, 10u64),
        (PIECE_LEN - 3, 6),
        (2 * PIECE_LEN, PIECE_LEN),
        (b.len() as u64 - 1, 1),
    ] {
        let got = store.get_range(&k, off, len).unwrap().unwrap();
        assert_eq!(got.as_ref(), &b[off as usize..(off + len) as usize]);
    }
    assert_eq!(
        store.get_range(&k, b.len() as u64, 1),
        Err(StoreError::OutOfRange)
    );

    // The table was written during the put and checks out against the κ.
    assert!(dir.0.join("trees").join(kappa_file_name(&k)).is_file());
    let tree = store.piece_tree(&k).unwrap().unwrap();
    assert_eq!(tree, PieceTree::from_bytes(&b));
    tree.verify_root(&k).unwrap();

    // Planted defect: a corrupted cached table is detected, discarded, and recomputed.
    std::fs::write(dir.0.join("trees").join(kappa_file_name(&k)), b"junk").unwrap();
    assert_eq!(store.piece_tree(&k).unwrap().unwrap(), tree);

    // Scrub: the blob verifies; after a bit flips on disk it does not.
    assert_eq!(store.verify_blob(&k).unwrap(), Some(true));
    let path = store.blob_path(&k);
    let mut on_disk = std::fs::read(&path).unwrap();
    on_disk[3 * PIECE_LEN as usize + 1] ^= 0x10;
    std::fs::write(&path, &on_disk).unwrap();
    assert_eq!(store.verify_blob(&k).unwrap(), Some(false));
    std::fs::remove_file(dir.0.join("trees").join(kappa_file_name(&k))).unwrap();
    assert_eq!(
        store.piece_tree(&k),
        Err(StoreError::KappaMismatch),
        "a table is never computed over bytes that no longer match their κ"
    );
}

#[test]
fn a_refused_streaming_put_leaves_nothing_behind() {
    let dir = TempDir::new("refuse");
    let store = FileKappaStore::open(&dir.0).unwrap();
    let b = blob(2 * PIECE_LEN as usize + 1, 3);
    let k = address_bytes(&b);
    // Planted defect: the source flips one byte in the middle of the stream.
    let mut bad = b.clone();
    bad[PIECE_LEN as usize] ^= 1;
    let mut w = store.put_writer("blake3", Some(&k)).unwrap();
    w.write(&bad).unwrap();
    assert_eq!(w.finish(), Err(StoreError::KappaMismatch));
    assert!(!store.contains(&k));
    assert!(!store.contains(&address_bytes(&bad)));
    assert!(tmp_is_empty(&dir));
    assert!(store.put_writer("sha256", None).is_err());
}

#[test]
fn partial_download_accepts_out_of_order_pieces_refuses_forgeries_and_resumes() {
    let dir = TempDir::new("partial");
    let b = blob(6 * PIECE_LEN as usize + 1234, 11);
    let k = address_bytes(&b);
    let tree = PieceTree::from_bytes(&b);
    let piece = |i: u64| {
        let r = tree.piece_range(i).unwrap();
        b[r.start as usize..r.end as usize].to_vec()
    };

    {
        let store = FileKappaStore::open(&dir.0).unwrap();
        // Planted defect: a table that does not hash to the κ cannot start a download.
        let other = PieceTree::from_bytes(&blob(b.len(), 12));
        assert_eq!(
            store.begin_partial(&k, other).err(),
            Some(StoreError::KappaMismatch)
        );

        let part = store.begin_partial(&k, tree.clone()).unwrap();
        assert_eq!(part.missing().len() as u64, piece_count(b.len() as u64));
        // Pieces from several threads, out of order.
        std::thread::scope(|s| {
            for i in [5u64, 1, 6] {
                let part = &part;
                let bytes = piece(i);
                s.spawn(move || part.put_piece(i, &bytes).unwrap());
            }
        });
        // Planted defect: a source sends piece 3 with one flipped bit.
        let mut forged = piece(3);
        forged[77] ^= 1;
        assert_eq!(part.put_piece(3, &forged), Err(StoreError::KappaMismatch));
        // ...or a correct piece under the wrong index.
        assert_eq!(part.put_piece(2, &piece(0)), Err(StoreError::KappaMismatch));
        assert_eq!(part.missing(), vec![0, 2, 3, 4]);
        assert!(
            part.commit().is_err(),
            "an incomplete download cannot install"
        );
        assert!(!store.contains(&k));
    }

    // Restart. Planted defect: piece 5 was damaged on disk while the process was down.
    let part_path = dir
        .0
        .join("partial")
        .join(format!("{}.part", kappa_file_name(&k)));
    let mut on_disk = std::fs::read(&part_path).unwrap();
    on_disk[5 * PIECE_LEN as usize + 9] ^= 0xff;
    std::fs::write(&part_path, &on_disk).unwrap();

    let store = FileKappaStore::open(&dir.0).unwrap();
    assert_eq!(store.partials(), vec![k]);
    let part = store.resume_partial(&k).unwrap().unwrap();
    assert_eq!(
        part.missing(),
        vec![0, 2, 3, 4, 5],
        "resume keeps verified pieces and re-fetches the damaged one"
    );
    for i in part.missing() {
        part.put_piece(i, &piece(i)).unwrap();
    }
    assert!(part.is_complete());
    assert_eq!(part.commit().unwrap(), k);
    assert_eq!(store.verify_blob(&k).unwrap(), Some(true));
    assert_eq!(store.piece_tree(&k).unwrap().unwrap(), tree);
    assert!(store.partials().is_empty());
    assert!(!part_path.exists());

    // Starting a download of content already held is a no-op that commits immediately.
    let again = store.begin_partial(&k, tree.clone()).unwrap();
    assert!(again.is_complete());
    assert_eq!(again.commit().unwrap(), k);
}

#[test]
fn single_piece_and_empty_blobs_work_through_every_path() {
    let dir = TempDir::new("small");
    let store = FileKappaStore::open(&dir.0).unwrap();
    for b in [Vec::new(), blob(1, 1), blob(PIECE_LEN as usize, 2)] {
        let k = address_bytes(&b);
        let tree = PieceTree::from_bytes(&b);
        let part = store.begin_partial(&k, tree).unwrap();
        part.put_piece(0, &b).unwrap();
        assert_eq!(part.commit().unwrap(), k);
        assert_eq!(store.get(&k).unwrap().unwrap().as_ref(), b.as_slice());
        assert_eq!(store.piece_tree(&k).unwrap().unwrap().piece_count(), 1);
    }
}
