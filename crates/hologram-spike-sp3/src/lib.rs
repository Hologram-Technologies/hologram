//! **P0.5 de-risk spike (D28)** — the composition bet, as runnable code.
//!
//! A minimal [`Client`] over the [`hologram_space::Space`] contract drives
//! `compile → store → boot`: a **synchronous** compile, a **synchronous** store, and the
//! **async** network/boot seam calling into the **synchronous** compute hot path. This is
//! the SP-3 witness — it proves the structural bet composes on native (the test below) and
//! `wasm32` (`cargo build --target wasm32-unknown-unknown`). The real `Client` supersedes
//! it in P1; until then it is kept, not thrown away.
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
use alloc::vec::Vec;
// `async_trait` desugars to `Box`ed futures; not in the `no_std` prelude.
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use hologram_space::{Bytes, KappaLabel71, KappaStore, Resolver, Space, StoreError};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I64: u8 = 5;

/// The smallest graph that actually computes: an i64→f32 cast of a rank-1 tensor.
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

/// The single programmatic surface (minimal P0.5 form): generic over any [`Space`], so it
/// is monomorphized per platform. Wires the synchronous compute path and the async
/// network/boot seam behind one type.
pub struct Client<S: Space> {
    space: S,
}

impl<S: Space> Client<S> {
    /// Build a client over a concrete space.
    pub fn new(space: S) -> Self {
        Self { space }
    }

    /// Compile a workload to `.holo` bytes — **synchronous** (pure compute hot path).
    pub fn compile(&self) -> Vec<u8> {
        let out =
            compile(cast_graph(), BackendKind::Cpu, WittLevel::W32).expect("compile cast graph");
        out.archive
    }

    /// Store the `.holo` in the space's **synchronous** [`KappaStore`], returning its κ.
    pub fn store_holo(&self, holo: &[u8]) -> Result<KappaLabel71, StoreError> {
        self.space.store().put("blake3", holo)
    }

    /// Boot a stored workload by κ — the **async network/boot seam**: try the resolver
    /// (network) first, fall back to the local synchronous store, then run the
    /// **synchronous** compute hot path. Returns the cast output values.
    ///
    /// This method is the composition proof: an `async fn` (the seam) that awaits the
    /// async resolver and then calls straight into synchronous store + compute — the only
    /// async↔sync transition (LAW-4).
    pub async fn boot(&self, kappa: &KappaLabel71, input: &[u8]) -> Vec<f32> {
        let holo: Bytes = match self.space.resolver().resolve(kappa).await.expect("resolve") {
            Some(bytes) => bytes,
            None => self
                .space
                .store()
                .get(kappa)
                .expect("store get")
                .expect("holo present locally"),
        };
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&holo, backend).expect("load session");
        let outputs = session
            .execute(&[InputBuffer { bytes: input }])
            .expect("execute");
        outputs[0]
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// A trivial concrete [`Space`] for the spike: a mock-engine [`Runtime`] over an in-memory
/// store, plus a resolver that never resolves (forcing the local-store fallback). It
/// demonstrates the *composition*, not the network — a real resolver arrives with
/// `hologram-net` in a later phase. The `Space::Store` is the runtime's own store, so
/// `store()` and `runtime()` share one content store (no duplication).
pub struct SpikeSpace {
    runtime: hologram_runtime::Runtime<hologram_runtime::MockEngine, hologram_space::MemKappaStore>,
    resolver: NullResolver,
}

impl SpikeSpace {
    /// A fresh spike space with a mock-engine runtime over an empty in-memory store.
    pub fn new() -> Self {
        Self {
            runtime: hologram_runtime::Runtime::new(
                hologram_runtime::MockEngine,
                hologram_space::MemKappaStore::new(),
            ),
            resolver: NullResolver,
        }
    }
}

impl Default for SpikeSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl Space for SpikeSpace {
    type Store = hologram_space::MemKappaStore;
    type Resolver = NullResolver;
    type Runtime =
        hologram_runtime::Runtime<hologram_runtime::MockEngine, hologram_space::MemKappaStore>;

    fn store(&self) -> &Self::Store {
        self.runtime.store()
    }
    fn resolver(&self) -> &Self::Resolver {
        &self.resolver
    }
    fn runtime(&self) -> &Self::Runtime {
        &self.runtime
    }
}

/// A [`Resolver`] that resolves nothing — the spike proves composition via the local
/// store, so the async seam simply returns `Ok(None)`.
pub struct NullResolver;

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl Resolver for NullResolver {
    async fn resolve(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(None)
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl Resolver for NullResolver {
    async fn resolve(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SP-3 on native: a `Space` composes async storage/boot with sync compute, and
    /// `Client` drives compile → store → boot end to end (i64→f32 cast).
    #[test]
    fn compile_store_boot_composes_on_native() {
        let client = Client::new(SpikeSpace::new());

        let holo = client.compile();
        let kappa = client.store_holo(&holo).expect("store");

        let vals: [i64; 4] = [0, 42, -7, 1024];
        let mut input = Vec::new();
        for &v in &vals {
            input.extend_from_slice(&v.to_le_bytes());
        }

        let got = pollster::block_on(client.boot(&kappa, &input));
        assert_eq!(got, alloc::vec![0.0, 42.0, -7.0, 1024.0]);
    }
}
