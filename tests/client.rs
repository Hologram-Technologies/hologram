//! `hologram::Client` (D4) end-to-end: the single programmatic surface drives
//! compile → provision → run over a minimal space (MemKappaStore + a null sync seam),
//! computing an i64→f32 cast. This is the SP-3 composition proof through the kept `Client`.
#![cfg(feature = "client")]

use hologram::graph::node::Node;
use hologram::graph::registry::{DTypeId, ShapeDescriptor};
use hologram::graph::{Graph, GraphOp, InputSource, OpKind};
use hologram::space::{
    Bytes, KappaLabel71, KappaSync, ManualClock, MemKappaStore, SeededEntropy, Space, SyncError,
};
use hologram::Client;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I64: u8 = 5;

/// The smallest graph that computes: an i64→f32 cast of a rank-1 tensor.
fn cast_graph() -> Graph {
    let mut graph = Graph::new();
    let sh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));
    let inp = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_I64),
        output_shape: sh,
    });
    graph.add_input(inp);
    let cast = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Cast),
        inputs: SmallVec::from_iter([InputSource::Node(inp)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cast)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_output(out);
    graph
}

/// A minimal space: a mock-engine `Runtime` over an in-memory store, plus a sync seam that
/// never fetches (forcing the local-store fallback) — enough to prove the Client
/// composition, not the network. `store()` delegates to the runtime's store, so `store()`
/// and `runtime()` share one content store.
struct TestSpace {
    runtime: hologram_runtime::Runtime<hologram_runtime::MockEngine, MemKappaStore>,
    sync: NullSync,
    entropy: SeededEntropy,
    clock: ManualClock,
}

impl TestSpace {
    fn new() -> Self {
        Self {
            runtime: hologram_runtime::Runtime::new(
                hologram_runtime::MockEngine,
                MemKappaStore::new(),
            ),
            sync: NullSync,
            entropy: SeededEntropy::default(),
            clock: ManualClock::default(),
        }
    }
}

impl Space for TestSpace {
    type Store = MemKappaStore;
    type Sync = NullSync;
    type Runtime = hologram_runtime::Runtime<hologram_runtime::MockEngine, MemKappaStore>;
    type Entropy = SeededEntropy;
    type Clock = ManualClock;

    fn store(&self) -> &Self::Store {
        self.runtime.store()
    }
    fn sync(&self) -> &Self::Sync {
        &self.sync
    }
    fn runtime(&self) -> &Self::Runtime {
        &self.runtime
    }
    fn entropy(&self) -> &Self::Entropy {
        &self.entropy
    }
    fn clock(&self) -> &Self::Clock {
        &self.clock
    }
}

struct NullSync;

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl KappaSync for NullSync {
    async fn fetch(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(None)
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

#[test]
fn client_compiles_provisions_and_runs_a_cast() {
    let client = Client::builder()
        .space(TestSpace::new())
        .build()
        .expect("build client");

    // compile (sync) → provision (sync store) → run (async seam → sync compute).
    let holo = client.compile(cast_graph()).expect("compile");
    let kappa = client.provision(&holo).expect("provision");

    // the provisioned κ is present in the local store.
    assert!(client.get(&kappa).expect("get").is_some());
    assert_eq!(client.ls(), vec![kappa]);
    assert!(client.verify(holo.as_bytes(), &kappa));

    let vals: [i64; 4] = [0, 42, -7, 1024];
    let mut input = Vec::new();
    for &v in &vals {
        input.extend_from_slice(&v.to_le_bytes());
    }
    let outputs = pollster::block_on(client.run(&kappa, &[input.as_slice()])).expect("run");

    let got: Vec<f32> = outputs[0]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![0.0, 42.0, -7.0, 1024.0]);
}

#[test]
fn client_opens_boots_and_suspends_a_session() {
    use hologram::space::{
        Capabilities, CapabilitySet, ContainerManifest, KappaStore, Realization,
    };
    use hologram_runtime::Phase;

    let client = Client::builder()
        .space(TestSpace::new())
        .build()
        .expect("build client");

    // Provision a container into the space's store: code + state + params → manifest κ (the
    // container id); a capability set → caps κ. (The mock engine accepts arbitrary code bytes.)
    let store = client.store();
    let code = store.put("blake3", b"<mock-code>").expect("code");
    let state = store.put("blake3", b"INIT").expect("state");
    let params = store.put("blake3", b"params").expect("params");
    let cid = store
        .put(
            "blake3",
            &ContainerManifest {
                code,
                initial_state: state,
                parameters: params,
            }
            .canonicalize(),
        )
        .expect("manifest");
    let caps = Capabilities {
        storage_roots: vec![],
        publish_channels: vec![],
        subscribe_channels: vec![],
        storage_quota_bytes: 1 << 16,
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: 0,
        network_fetch: false,
        network_announce: false,
    };
    let caps_k = store
        .put("blake3", &CapabilitySet::new(caps).canonicalize())
        .expect("caps");

    // open → boot → suspend, driven over the space's runtime (MockEngine).
    let mut session = client.open(&cid, &caps_k);
    assert_eq!(session.phase(), Phase::Provisioned);
    pollster::block_on(session.boot()).expect("boot");
    assert_eq!(session.phase(), Phase::Running);
    let snapshot = pollster::block_on(session.suspend()).expect("suspend");
    assert_eq!(session.phase(), Phase::Suspended);
    assert_eq!(session.snapshot(), Some(&snapshot));
}
