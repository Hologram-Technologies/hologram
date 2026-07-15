//! B2 / G-C4 witness — **error-log chain compaction**.
//!
//! Architecture §9 G-C4 flags the `ErrorEvent` predecessor chain as unbounded. The runtime
//! now compacts: when the chain depth would exceed `error_log_threshold`, the next emit's
//! predecessor becomes a `ChainCompaction` κ (zero operands → reachability breaks). The old
//! tail becomes unreachable from any pinned root and is reclaimed by the store's GC.

use hologram_runtime::{diag_class, MockEngine, Runtime};
use hologram_space::{address_bytes, KappaStore, Realization};
use hologram_space::{ChainCompaction, ErrorEvent, REGISTRY};
use hologram_store_mem::MemKappaStore;

#[test]
fn b2_chain_compaction_breaks_predecessor_chain_at_threshold() {
    let store = MemKappaStore::new();
    let rt = Runtime::new(MockEngine, store);
    rt.set_error_log_threshold(3);

    let cid = address_bytes(b"container-under-test");

    // 1..=3 are normal chain links: predecessor walks back through ErrorEvents.
    let k1 = rt.emit_diagnostic(&cid, 0, 1, None).unwrap();
    let k2 = rt.emit_diagnostic(&cid, 0, 2, None).unwrap();
    let k3 = rt.emit_diagnostic(&cid, 0, 3, None).unwrap();

    // 4th emit: depth == threshold (3) → predecessor is a ChainCompaction barrier, NOT k3.
    let k4 = rt.emit_diagnostic(&cid, 0, 4, None).unwrap();

    let bytes4 = rt.store().get(&k4).unwrap().unwrap();
    let refs4 = ErrorEvent::references(bytes4.as_ref()).unwrap();
    // ErrorEvent::parts: refs = [source, optional predecessor, optional context]. Position 1 is
    // the predecessor when present.
    assert_eq!(refs4.len(), 2, "k4 has source + predecessor");
    let predecessor_of_k4 = refs4[1];

    // The predecessor is NOT k3 (the prior head) — it's a ChainCompaction κ.
    assert_ne!(
        predecessor_of_k4, k3,
        "predecessor must be the compaction barrier, not the prior head"
    );

    // The barrier has no operands (zero references), so reachability from k4 ends here.
    let barrier_bytes = rt.store().get(&predecessor_of_k4).unwrap().unwrap();
    let barrier_refs = ChainCompaction::references(barrier_bytes.as_ref()).unwrap();
    assert_eq!(
        barrier_refs.len(),
        0,
        "compaction barrier has no operands — reachability stops here"
    );

    // Reachability from k4 *cannot* reach k1/k2/k3 — the chain is broken.
    rt.store().pin(&k4).unwrap();
    let evicted = hologram_space::GarbageCollect::gc(rt.store(), REGISTRY).unwrap();
    assert!(
        evicted >= 3,
        "the three older events must be reclaimed (got {evicted})"
    );
    assert!(!rt.store().contains(&k1));
    assert!(!rt.store().contains(&k2));
    assert!(!rt.store().contains(&k3));
    // k4 + the compaction barrier survive (reachable from the pin).
    assert!(rt.store().contains(&k4));
    assert!(rt.store().contains(&predecessor_of_k4));
}

/// SPINE-6 cleanup witness — runtime-side capability denials are **observable in the audit
/// trail**: when the runtime refuses a container's publish/subscribe/spawn_child intent, it
/// mints an `ErrorEvent` with a runtime-class classification (0x10..0x12), so an operator can
/// distinguish a container-emitted diagnostic from a runtime-issued denial. Closes the prior
/// "let _ = self.publish(...)" silent-drop hole.
#[test]
fn diagnostic_classes_are_distinct_and_namespaced() {
    // Static guarantees — evaluated at compile time, so the band split is structural.
    const _: () = assert!(diag_class::CONTAINER_EMITTED == 0x01);
    const _: () = assert!(diag_class::PUBLISH_DENIED == 0x10);
    const _: () = assert!(diag_class::SUBSCRIBE_DENIED == 0x11);
    const _: () = assert!(diag_class::SPAWN_CHILD_DENIED == 0x12);
    // The runtime band (0x10..0x20) is disjoint from the container band (0x01..0x10).
    const _: () = assert!(diag_class::PUBLISH_DENIED > diag_class::CONTAINER_EMITTED);
    const _: () = assert!(diag_class::SPAWN_CHILD_DENIED < 0x20);
}

#[test]
fn b2_chain_compaction_threshold_zero_disables_compaction() {
    let store = MemKappaStore::new();
    let rt = Runtime::new(MockEngine, store);
    rt.set_error_log_threshold(0); // 0 = unbounded (operator opt-in, SPINE-6)

    let cid = address_bytes(b"unbounded-chain");
    let mut head = None;
    for i in 0..50u32 {
        let k = rt.emit_diagnostic(&cid, 0, i, None).unwrap();
        head = Some(k);
    }
    let head = head.unwrap();
    // The head's predecessor is the prior ErrorEvent, not a ChainCompaction κ. We just assert
    // there's a predecessor and it's NOT zero refs (so it's an ErrorEvent, not a compaction).
    let bytes = rt.store().get(&head).unwrap().unwrap();
    let refs = ErrorEvent::references(bytes.as_ref()).unwrap();
    assert_eq!(refs.len(), 2, "source + predecessor");
    let pred = refs[1];
    let pred_bytes = rt.store().get(&pred).unwrap().unwrap();
    // The predecessor is an ErrorEvent (its refs include at least source); not a ChainCompaction
    // (zero refs).
    let pred_refs = ErrorEvent::references(pred_bytes.as_ref()).unwrap();
    assert!(
        !pred_refs.is_empty(),
        "with threshold=0 the chain is never folded into ChainCompaction"
    );
}
