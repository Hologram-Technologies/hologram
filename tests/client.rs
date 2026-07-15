//! `hologram::Client` (D4) end-to-end: the single programmatic surface drives
//! compile → provision → run over a minimal space (MemKappaStore + a null resolver),
//! computing an i64→f32 cast. This is the SP-3 composition proof through the kept `Client`.
#![cfg(feature = "client")]

use hologram::graph::node::Node;
use hologram::graph::registry::{DTypeId, ShapeDescriptor};
use hologram::graph::{Graph, GraphOp, InputSource, OpKind};
use hologram::space::{Bytes, KappaLabel71, MemKappaStore, Resolver, Space, StoreError};
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

/// A minimal space: an in-memory store plus a resolver that never resolves (forcing the
/// local-store fallback) — enough to prove the Client composition, not the network.
struct TestSpace {
    store: MemKappaStore,
    resolver: NullResolver,
}

impl Space for TestSpace {
    type Store = MemKappaStore;
    type Resolver = NullResolver;

    fn store(&self) -> &Self::Store {
        &self.store
    }
    fn resolver(&self) -> &Self::Resolver {
        &self.resolver
    }
}

struct NullResolver;

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Resolver for NullResolver {
    async fn resolve(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(None)
    }
}

#[test]
fn client_compiles_provisions_and_runs_a_cast() {
    let space = TestSpace {
        store: MemKappaStore::new(),
        resolver: NullResolver,
    };
    let client = Client::builder()
        .space(space)
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
