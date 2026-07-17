//! # hologram-net-bare
//!
//! Bare-metal [`KappaSync`] (architecture §2 / C2). Symmetric to `hologram-net-http` /
//! `hologram-net-tcp` for std hosts, but no_std: the substrate's network surface over the HAL
//! [`NetworkInterface`] trait. **No filesystem, no OS-level sockets** — only a frame-oriented
//! NIC and a small frame codec.
//!
//! ## What this crate provides
//! - [`BareNetSync`] — a [`KappaSync`] driving a `NetworkInterface` directly. Implements the
//!   uor-native fetch/announce/discover surface over the simplest possible wire format:
//!   length-prefixed CBOR-ish frames (`u32 LE len | u8 kind | payload`). Verify-on-receipt at
//!   every fetch (SPINE-4).
//! - A minimal **frame codec**: REQ/RES kinds for `fetch`, `announce`, `discover`.
//! - A **peer table** the substrate populates from boot-time hardware-inventory + per-peer MAC.
//!
//! ## Wire compatibility with `hologram-net-tcp`
//! The frame format (length-prefixed `u32 LE len | u8 kind | payload`, append-only kinds) is the
//! same shape used by `hologram-net-tcp` on std hosts. Peer identity is κ in both crates (the κ
//! of a `PeerEndpoint` realization); there are no PeerIds or Multiaddrs in either. A bare-metal
//! node and a std node speak the same uor-native protocol — no libp2p layer on either side.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use async_trait::async_trait;
use core::future::poll_fn;
use core::task::Poll;
use hashbrown::{HashMap, HashSet};
use hologram_space::NetworkInterface;
use hologram_space::{verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaSync, SyncError};
use spin::Mutex;

use crate::protocol::WireVersionRange;

// ── frame codec ─────────────────────────────────────────────────────────────

/// Frame kinds on the wire. Append-only — never renumber an existing kind (SPINE-5).
/// The connect-handshake HELLO carrying a `WireVersionRange` (spec 04 §Protocol hardening).
const KIND_HELLO: u8 = 0x00;
const KIND_FETCH_REQ: u8 = 0x01;
const KIND_FETCH_RES_OK: u8 = 0x02;
const KIND_FETCH_RES_404: u8 = 0x03;
const KIND_ANNOUNCE: u8 = 0x10;
const KIND_DISCOVER_REQ: u8 = 0x20;
const KIND_DISCOVER_RES: u8 = 0x21;

/// Build an outbound frame: `u32 LE len | u8 kind | payload`.
fn encode_frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    let len = (1 + payload.len()) as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

/// Parse an inbound frame; returns `(kind, payload, total_bytes_consumed)`.
pub fn decode_frame(buf: &[u8]) -> Option<(u8, &[u8], usize)> {
    if buf.len() < 5 {
        return None;
    }
    let len = u32::from_le_bytes(buf[..4].try_into().ok()?) as usize;
    if buf.len() < 4 + len || len < 1 {
        return None;
    }
    let kind = buf[4];
    let payload = &buf[5..4 + len];
    Some((kind, payload, 4 + len))
}

// ── connect handshake (wire-version negotiation, spec 04 §Protocol hardening) ──

/// Why the connect handshake failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeError {
    /// The peer's first frame was not a well-formed HELLO (wrong kind, or a malformed range).
    BadHello,
    /// No wire version both peers support — they are incompatible; refuse (never a silent downgrade).
    Incompatible,
}

/// Build the connect-handshake HELLO frame advertising `range`. Each peer sends this first on a new
/// connection; the frame is a normal `len | KIND_HELLO | min:u16 ‖ max:u16` frame.
#[must_use]
pub fn hello_frame(range: WireVersionRange) -> Vec<u8> {
    encode_frame(KIND_HELLO, &range.encode())
}

/// Given our advertised `local` range and the peer's inbound HELLO frame bytes, negotiate the wire
/// version to speak — or refuse (the receiver half of the handshake; the sender emits [`hello_frame`]
/// first). Never panics on hostile bytes.
pub fn negotiate_from_hello(
    local: WireVersionRange,
    peer_hello: &[u8],
) -> Result<u16, HandshakeError> {
    let (kind, payload, _n) = decode_frame(peer_hello).ok_or(HandshakeError::BadHello)?;
    if kind != KIND_HELLO {
        return Err(HandshakeError::BadHello);
    }
    let peer = WireVersionRange::decode(payload).ok_or(HandshakeError::BadHello)?;
    local.negotiate(peer).ok_or(HandshakeError::Incompatible)
}

// ── BareNetSync ─────────────────────────────────────────────────────────────

/// Resolver hook: given a κ, produce the canonical bytes if locally available. The bare-metal
/// substrate wires this to its `KappaStore::get` so a `BareNetSync` can answer inbound
/// fetch requests with locally-stored content (verify-on-receipt happens on the *receiver*'s
/// side; the responder just sends what it has).
pub type LocalResolver = Arc<dyn Fn(&KappaLabel71) -> Option<Bytes> + Send + Sync>;

/// Discovery hook: lists locally-stored top-level κs.
pub type LocalIterator = Arc<dyn Fn() -> Vec<KappaLabel71> + Send + Sync>;

/// Bare-metal [`KappaSync`] over a [`NetworkInterface`].
///
/// The network surface here is genuinely uor-native: every fetched byte is re-derived through
/// the σ-axis at the receiver (SPINE-4); a forging responder is rejected. Announce/discover
/// are best-effort frames over the NIC; the wire format is forward-compatible (frame kinds are
/// append-only).
///
/// **Wait discipline (no arbitrary poll caps).** `fetch` registers a task waker with the NIC
/// (`register_rx_waker`) and suspends until the driver fires `notify_rx`. Each notification
/// drains all pending frames; if the expected response has been delivered, `fetch` resumes. No
/// hardcoded retry count — the bound on wait time is the caller's own timeout discipline (or
/// none — async drops are clean).
pub struct BareNetSync {
    nic: Arc<dyn NetworkInterface>,
    local_get: LocalResolver,
    local_iter: LocalIterator,
    /// Pending inbound bytes from the NIC, awaiting frame boundary.
    rx_buf: Mutex<Vec<u8>>,
    /// Pending fetch responses keyed by requested κ. The poll loop populates; `fetch` drains.
    fetch_results: Mutex<HashMap<[u8; 71], Option<Bytes>>>,
    /// Discovered κs from inbound DISCOVER_RES frames — populated by `poll()`, drained by
    /// `discover()`. Dedup is by content-address (HashSet).
    discovered: Mutex<HashSet<[u8; 71]>>,
}

impl BareNetSync {
    pub fn new(
        nic: Arc<dyn NetworkInterface>,
        local_get: LocalResolver,
        local_iter: LocalIterator,
    ) -> Self {
        Self {
            nic,
            local_get,
            local_iter,
            rx_buf: Mutex::new(Vec::new()),
            fetch_results: Mutex::new(HashMap::new()),
            discovered: Mutex::new(HashSet::new()),
        }
    }

    /// Drain inbound frames from the NIC; respond to incoming requests; record fetch responses.
    /// Production deployment drives this from the NIC's `register_rx_waker` callback (one tick
    /// per RX-ready signal). Returns the number of frames processed.
    pub fn poll(&self) -> Result<usize, SyncError> {
        let mtu = self.nic.mtu() as usize;
        let mut frame = alloc::vec![0u8; mtu];
        let mut processed = 0usize;
        loop {
            let n = self
                .nic
                .receive(&mut frame)
                .map_err(|_| SyncError::BackendFailure("nic-rx"))?;
            if n == 0 {
                break;
            }
            self.rx_buf.lock().extend_from_slice(&frame[..n]);
            processed += 1;
        }
        // Consume complete frames out of rx_buf.
        loop {
            let consumed = {
                let buf = self.rx_buf.lock();
                let Some((kind, payload, total)) = decode_frame(&buf) else {
                    break;
                };
                self.handle_frame(kind, payload)?;
                total
            };
            self.rx_buf.lock().drain(..consumed);
        }
        Ok(processed)
    }

    fn handle_frame(&self, kind: u8, payload: &[u8]) -> Result<(), SyncError> {
        match kind {
            KIND_FETCH_REQ => {
                if payload.len() != 71 {
                    return Ok(()); // ignore malformed
                }
                let mut k = [0u8; 71];
                k.copy_from_slice(payload);
                let label =
                    KappaLabel::from_bytes(&k).map_err(|_| SyncError::VerificationFailed)?;
                if let Some(bytes) = (self.local_get)(&label) {
                    let mut buf = Vec::with_capacity(71 + bytes.len());
                    buf.extend_from_slice(&k);
                    buf.extend_from_slice(bytes.as_ref());
                    self.send_frame(KIND_FETCH_RES_OK, &buf)?;
                } else {
                    self.send_frame(KIND_FETCH_RES_404, payload)?;
                }
            }
            KIND_FETCH_RES_OK => {
                if payload.len() < 71 {
                    return Ok(());
                }
                let mut k = [0u8; 71];
                k.copy_from_slice(&payload[..71]);
                let label =
                    KappaLabel::from_bytes(&k).map_err(|_| SyncError::VerificationFailed)?;
                let bytes = &payload[71..];
                // SPINE-4 — verify on receipt. A forging responder is rejected.
                if verify_kappa(bytes, &label) == Ok(true) {
                    let arc: Bytes = Arc::<[u8]>::from(bytes);
                    self.fetch_results.lock().insert(k, Some(arc));
                } else {
                    self.fetch_results.lock().insert(k, None);
                }
            }
            KIND_FETCH_RES_404 if payload.len() == 71 => {
                let mut k = [0u8; 71];
                k.copy_from_slice(payload);
                self.fetch_results.lock().insert(k, None);
            }
            KIND_DISCOVER_REQ => {
                // Reply with as many locally-iterated κs as fit in one MTU. The MTU bound is the
                // structural cap (the NIC's own frame-size limit), not an arbitrary policy cap.
                let listed = (self.local_iter)();
                let mut payload = Vec::with_capacity(4 + listed.len() * 71);
                let mtu_cap = (self.nic.mtu() as usize - 4 - 1 - 4) / 71; // -4 len -1 kind -4 count
                let n = mtu_cap.min(listed.len());
                payload.extend_from_slice(&(n as u32).to_le_bytes());
                for k in listed.iter().take(n) {
                    payload.extend_from_slice(k.as_array());
                }
                self.send_frame(KIND_DISCOVER_RES, &payload)?;
            }
            KIND_DISCOVER_RES => {
                if payload.len() < 4 {
                    return Ok(());
                }
                let n = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
                let mut off = 4;
                let mut found = self.discovered.lock();
                for _ in 0..n {
                    if off + 71 > payload.len() {
                        break;
                    }
                    let mut k = [0u8; 71];
                    k.copy_from_slice(&payload[off..off + 71]);
                    // Validate parse before storing (forged κs that don't parse are silently
                    // dropped; verify-on-receipt is for fetched *content*, not for advertised
                    // κs which are merely hints).
                    if KappaLabel::<71>::from_bytes(&k).is_ok() {
                        found.insert(k);
                    }
                    off += 71;
                }
            }
            _ => {} // unknown kinds are ignored — forward-compat (SPINE-5)
        }
        Ok(())
    }

    fn send_frame(&self, kind: u8, payload: &[u8]) -> Result<(), SyncError> {
        let frame = encode_frame(kind, payload);
        self.nic
            .transmit(&frame)
            .map_err(|_| SyncError::BackendFailure("nic-tx"))?;
        Ok(())
    }
}

// Maybe-Send follow-through (LAW-4): the [`KappaSync`] trait is `Send + Sync` on native/bare-arm
// and `?Send` on `wasm32`. `#[async_trait]` and `#[async_trait(?Send)]` desugar to *different*
// future types, so the impl attribute must track the trait per target (else E0053).
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl KappaSync for BareNetSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        // Local hit short-circuits without a network round-trip.
        if let Some(b) = (self.local_get)(kappa) {
            return Ok(Some(b));
        }
        self.send_frame(KIND_FETCH_REQ, kappa.as_array())?;
        // Waker-based wait — no hardcoded poll count. The driver fires `notify_rx` on each
        // inbound frame; `poll()` drains pending bytes; if the expected response has been
        // recorded, resume. If the caller wants a timeout, they wrap the future themselves —
        // this method's wait is bounded by the network, not by a policy constant (SPINE-6).
        let key = *kappa.as_array();
        poll_fn(|cx| {
            // 1. Drain whatever the NIC may have buffered between calls.
            if let Err(e) = self.poll() {
                return Poll::Ready(Err(e));
            }
            if let Some(v) = self.fetch_results.lock().remove(&key) {
                return Poll::Ready(Ok(v));
            }
            // 2. Register for the next RX-ready notification (lost-wakeup-safe: if a frame
            //    arrived between drain and register, the next register_rx_waker call wakes
            //    immediately, per the NIC trait's contract).
            self.nic.register_rx_waker(cx.waker().clone());
            // 3. Race: re-check after registration so a notification that fired between
            //    drain and register is observed without losing the wakeup.
            if let Err(e) = self.poll() {
                return Poll::Ready(Err(e));
            }
            if let Some(v) = self.fetch_results.lock().remove(&key) {
                return Poll::Ready(Ok(v));
            }
            Poll::Pending
        })
        .await
    }

    async fn announce(&self, kappa: &KappaLabel71) {
        let _ = self.send_frame(KIND_ANNOUNCE, kappa.as_array());
    }

    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        // Broadcast a discover request and drain whatever's already arrived. Real deployments
        // run `poll()` continuously off the RX waker so `discovered` accumulates new κs in the
        // background; `discover()` is a snapshot of that current knowledge. No retry loop and
        // no hardcoded delay — knowledge accumulates asynchronously, the caller polls when
        // they want a fresh view (SPINE-6: caller's pace, not policy's).
        let _ = self.send_frame(KIND_DISCOVER_REQ, &[]);
        let _ = self.poll();
        let found = self.discovered.lock();
        found
            .iter()
            .filter_map(|arr| KappaLabel::<71>::from_bytes(arr).ok())
            .collect()
    }

    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        // Bare-metal peers are bound at boot via the `HardwareInventory` realization (NIC MAC +
        // driver κ); there is no `host:port` resolution layer here. Fail-loud rather than silently
        // accept an unenforceable parameter (SPINE-6).
        Err(SyncError::NotEnabled)
    }

    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        // No HTTP URL surface on bare-metal — same reasoning as `add_peer`. Fail-loud.
        Err(SyncError::NotEnabled)
    }
}

impl fmt::Debug for BareNetSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BareNetSync")
            .field("nic_mac", &self.nic.mac_address())
            .field("nic_mtu", &self.nic.mtu())
            .finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use core::task::Waker;
    use hologram_space::KappaStore;
    use hologram_space::NicError;
    use hologram_tck::MemKappaStore;

    #[test]
    fn connect_handshake_negotiates_or_refuses() {
        let a = WireVersionRange { min: 1, max: 3 };
        let b = WireVersionRange { min: 2, max: 5 };
        // Each peer negotiates from the other's HELLO → the highest common version (symmetric).
        assert_eq!(negotiate_from_hello(b, &hello_frame(a)), Ok(3));
        assert_eq!(negotiate_from_hello(a, &hello_frame(b)), Ok(3));
        // An incompatible peer is refused — never a silent downgrade.
        let far = WireVersionRange { min: 9, max: 9 };
        assert_eq!(
            negotiate_from_hello(far, &hello_frame(a)),
            Err(HandshakeError::Incompatible)
        );
        // A non-HELLO first frame, or hostile garbage, is a clean BadHello (never a panic).
        assert_eq!(
            negotiate_from_hello(a, &encode_frame(KIND_ANNOUNCE, b"x")),
            Err(HandshakeError::BadHello)
        );
        assert_eq!(
            negotiate_from_hello(a, b"\x01"),
            Err(HandshakeError::BadHello)
        );
        assert_eq!(negotiate_from_hello(a, &[]), Err(HandshakeError::BadHello));
    }

    /// A loopback NIC: every `transmit` becomes available to the same NIC's `receive`. Backed
    /// by an internal queue — the simplest possible no_std-compatible NIC test fixture.
    struct LoopbackNic {
        mac: [u8; 6],
        mtu: u32,
        queue: Mutex<Vec<u8>>,
    }
    impl LoopbackNic {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                mac: [0x02, 0, 0, 0, 0, 1],
                mtu: 1500,
                queue: Mutex::new(Vec::new()),
            })
        }
    }
    impl NetworkInterface for LoopbackNic {
        fn mac_address(&self) -> [u8; 6] {
            self.mac
        }
        fn mtu(&self) -> u32 {
            self.mtu
        }
        fn transmit(&self, frame: &[u8]) -> Result<usize, NicError> {
            self.queue.lock().extend_from_slice(frame);
            Ok(frame.len())
        }
        fn receive(&self, buffer: &mut [u8]) -> Result<usize, NicError> {
            let mut q = self.queue.lock();
            let n = q.len().min(buffer.len());
            buffer[..n].copy_from_slice(&q[..n]);
            q.drain(..n);
            Ok(n)
        }
        fn register_rx_waker(&self, _waker: Waker) {}
    }

    #[test]
    fn frame_codec_round_trips() {
        let f = encode_frame(KIND_ANNOUNCE, b"hello");
        let (k, p, n) = decode_frame(&f).unwrap();
        assert_eq!(k, KIND_ANNOUNCE);
        assert_eq!(p, b"hello");
        assert_eq!(n, f.len());
    }

    #[test]
    fn bare_net_sync_fetches_from_self_via_nic_loopback() {
        pollster::block_on(async {
            let store = Arc::new(MemKappaStore::new());
            let payload = b"bare-metal-cas-payload";
            let k = store.put("blake3", payload).unwrap();

            let nic = LoopbackNic::new();
            let store_get = store.clone();
            let store_iter = store.clone();
            let sync = BareNetSync::new(
                nic.clone() as Arc<dyn NetworkInterface>,
                Arc::new(move |k: &KappaLabel71| store_get.get(k).ok().flatten()),
                Arc::new(move || store_iter.iterate()),
            );
            // Local hit short-circuits.
            let got = sync.fetch(&k).await.unwrap().unwrap();
            assert_eq!(got.as_ref(), payload);

            // Re-derive verification of the same κ over the wire path (fake-remote scenario):
            // we drop the local resolver to None and pre-populate the loopback queue with a
            // FETCH_RES_OK frame; verify-on-receipt should accept.
            let other_store = Arc::new(MemKappaStore::new());
            let other_iter = other_store.clone();
            let sync_ro = BareNetSync::new(
                nic.clone() as Arc<dyn NetworkInterface>,
                Arc::new(|_| None),
                Arc::new(move || other_iter.iterate()),
            );
            // Inject a FETCH_RES_OK frame (the sender side) so poll() consumes it as a response.
            let mut resp_payload = Vec::with_capacity(71 + payload.len());
            resp_payload.extend_from_slice(k.as_array());
            resp_payload.extend_from_slice(payload);
            let f = encode_frame(KIND_FETCH_RES_OK, &resp_payload);
            nic.transmit(&f).unwrap();
            sync_ro.poll().unwrap();
            // After poll, the fetch results table has the κ resolved.
            let opt = sync_ro
                .fetch_results
                .lock()
                .get(k.as_array())
                .cloned()
                .flatten()
                .unwrap();
            assert_eq!(opt.as_ref(), payload);
        });
    }

    #[test]
    fn bare_net_sync_rejects_forged_response() {
        pollster::block_on(async {
            let payload = b"truth";
            let truthful_k = hologram_space::address_bytes(payload);
            let nic = LoopbackNic::new();
            let sync = BareNetSync::new(
                nic.clone() as Arc<dyn NetworkInterface>,
                Arc::new(|_| None),
                Arc::new(|| alloc::vec![]),
            );
            // Build a FETCH_RES_OK frame that *claims* `truthful_k` but ships `forged` bytes.
            let mut payload = Vec::with_capacity(71 + 8);
            payload.extend_from_slice(truthful_k.as_array());
            payload.extend_from_slice(b"forgedXX");
            let f = encode_frame(KIND_FETCH_RES_OK, &payload);
            nic.transmit(&f).unwrap();
            sync.poll().unwrap();
            // SPINE-4: the receiver verifies — forged content is recorded as None (rejected).
            let recorded = sync
                .fetch_results
                .lock()
                .get(truthful_k.as_array())
                .cloned();
            assert!(matches!(recorded, Some(None)));
        });
    }

    /// Two NICs joined by crossed queues — one NIC's `transmit` is the other's `receive` and vice
    /// versa: an in-process point-to-point link (the loopback transport, no sockets). The simplest
    /// deterministic fixture for two-peer protocol tests.
    struct PairedNic {
        mac: [u8; 6],
        mtu: u32,
        tx: Arc<Mutex<Vec<u8>>>,
        rx: Arc<Mutex<Vec<u8>>>,
    }
    impl PairedNic {
        fn pair() -> (Arc<Self>, Arc<Self>) {
            let ab = Arc::new(Mutex::new(Vec::new()));
            let ba = Arc::new(Mutex::new(Vec::new()));
            let a = Arc::new(Self {
                mac: [0x02, 0, 0, 0, 0, 1],
                mtu: 1500,
                tx: ab.clone(),
                rx: ba.clone(),
            });
            let b = Arc::new(Self {
                mac: [0x02, 0, 0, 0, 0, 2],
                mtu: 1500,
                tx: ba,
                rx: ab,
            });
            (a, b)
        }
    }
    impl NetworkInterface for PairedNic {
        fn mac_address(&self) -> [u8; 6] {
            self.mac
        }
        fn mtu(&self) -> u32 {
            self.mtu
        }
        fn transmit(&self, frame: &[u8]) -> Result<usize, NicError> {
            self.tx.lock().extend_from_slice(frame);
            Ok(frame.len())
        }
        fn receive(&self, buffer: &mut [u8]) -> Result<usize, NicError> {
            let mut q = self.rx.lock();
            let n = q.len().min(buffer.len());
            buffer[..n].copy_from_slice(&q[..n]);
            q.drain(..n);
            Ok(n)
        }
        fn register_rx_waker(&self, _waker: Waker) {}
    }

    fn peer(nic: Arc<PairedNic>, store: Arc<MemKappaStore>) -> BareNetSync {
        let get = store.clone();
        let iter = store.clone();
        BareNetSync::new(
            nic as Arc<dyn NetworkInterface>,
            Arc::new(move |k: &KappaLabel71| get.get(k).ok().flatten()),
            Arc::new(move || iter.iterate()),
        )
    }

    #[test]
    fn two_peers_fetch_over_the_loopback_link() {
        // Peer A holds content; peer B (empty store) fetches it over an in-process link — the full
        // request/response transport path (no local shortcut), with verify-on-receipt.
        let a_store = Arc::new(MemKappaStore::new());
        let payload = b"content-that-only-peer-A-holds";
        let k = a_store.put("blake3", payload).unwrap();

        let (nic_a, nic_b) = PairedNic::pair();
        let peer_a = peer(nic_a, a_store);
        let peer_b = peer(nic_b.clone(), Arc::new(MemKappaStore::new()));

        // B sends a FETCH_REQ for `k` over the link (the request half of `fetch`).
        nic_b
            .transmit(&encode_frame(KIND_FETCH_REQ, k.as_array()))
            .unwrap();
        // A processes the request → resolves `k` → transmits FETCH_RES_OK back over the link.
        peer_a.poll().unwrap();
        // B processes the response → verify-on-receipt → records the content.
        peer_b.poll().unwrap();

        let got = peer_b
            .fetch_results
            .lock()
            .get(k.as_array())
            .cloned()
            .flatten()
            .expect("B resolved A's content over the link");
        assert_eq!(got.as_ref(), payload);
    }

    #[test]
    fn two_peers_fetch_miss_yields_404_over_the_link() {
        // B fetches a κ neither peer holds → A answers FETCH_RES_404 → B records the miss (not a hang).
        let (nic_a, nic_b) = PairedNic::pair();
        let peer_a = peer(nic_a, Arc::new(MemKappaStore::new()));
        let peer_b = peer(nic_b.clone(), Arc::new(MemKappaStore::new()));

        let absent = hologram_space::address_bytes(b"nobody-has-this");
        nic_b
            .transmit(&encode_frame(KIND_FETCH_REQ, absent.as_array()))
            .unwrap();
        peer_a.poll().unwrap();
        peer_b.poll().unwrap();

        // The miss is recorded as an explicit `None` (resolved-absent), not left pending.
        let recorded = peer_b.fetch_results.lock().get(absent.as_array()).cloned();
        assert_eq!(recorded, Some(None));
    }
}
