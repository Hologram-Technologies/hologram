//! The κ-Distribution `/v2/` registry server (spec 003).
//!
//! Its own `std::net` accept loop dispatches **Level 3** edge routes (`/v2/{path}/edges/…`) to this
//! crate's handlers — which consume the `uor-distribution` standard — and **delegates Levels 1-2**
//! (blobs, tags) to [`hologram_net::http::kd::handle_v2`]. One `/v2/` surface layered across the
//! publish boundary: the published `hologram-net` owns L1/L2; this leaf crate adds L3+ using the
//! unpublished standard. (When the standard is published at ratification, the server consolidates.)

use std::io::Read;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use hologram_space::KappaStore;

const MAX_HEADER_BYTES: usize = 64 * 1024;

/// A running registry server bound to a localhost port. Drop or call [`RegistryServer::shutdown`] to
/// stop it.
pub struct RegistryServer {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RegistryServer {
    /// The bound address.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// Stop the server and join its thread.
    pub fn shutdown(mut self) {
        self.stop();
    }
    fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr); // nudge the accept loop
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RegistryServer {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop();
        }
    }
}

/// Serve the κ-Distribution `/v2/` registry from `store` on an ephemeral localhost port (tests).
pub fn serve(store: Arc<dyn KappaStore>) -> std::io::Result<RegistryServer> {
    serve_addr(store, "127.0.0.1:0")
}

/// Serve the κ-Distribution `/v2/` registry from `store` on `addr`.
pub fn serve_addr(store: Arc<dyn KappaStore>, addr: &str) -> std::io::Result<RegistryServer> {
    let listener = TcpListener::bind(addr)?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let handle = std::thread::spawn(move || {
        for conn in listener.incoming() {
            if flag.load(Ordering::SeqCst) {
                break;
            }
            if let Ok(mut stream) = conn {
                let _ = handle_conn(&mut stream, store.as_ref());
            }
        }
    });
    Ok(RegistryServer {
        addr,
        shutdown,
        handle: Some(handle),
    })
}

fn handle_conn(stream: &mut TcpStream, store: &dyn KappaStore) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > MAX_HEADER_BYTES {
            break;
        }
    }
    let req = String::from_utf8_lossy(&buf);
    let path = req.split_whitespace().nth(1).unwrap_or("");
    let path_only = path.split('?').next().unwrap_or(path);
    // Levels 3-5 are served here; Levels 1-2 (blobs, tags, uploads, health, version) delegate to
    // hologram-net's kd binding — with our admission predicate so registered filters are enforced on
    // the blob-write path (spec §5.1, §10).
    let admit = |scope: &str, body: &[u8]| crate::filter::admit(scope, body);
    if !path_only.starts_with("/v2/") {
        return hologram_net::http::kd::handle_v2_admitted(stream, &buf, store, Some(&admit));
    }
    if path_only.contains("/compose/") {
        crate::compose::handle_compose(stream, &buf, store)
    } else if path_only.contains("/witnesses/") {
        crate::compose::handle_witness(stream, &buf, store)
    } else if path_only.contains("/schemas/") {
        crate::schema::handle_schema(stream, &buf, store)
    } else if path_only.contains("/edges") {
        crate::edge::handle_edges(stream, &buf, store)
    } else if path_only.contains("/gc/") {
        crate::gc::handle_gc(stream, &buf, store)
    } else if path_only.contains("/filters") {
        crate::filter::handle_filters(stream, &buf, store)
    } else {
        hologram_net::http::kd::handle_v2_admitted(stream, &buf, store, Some(&admit))
    }
}
