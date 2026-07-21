//! Level 5 — garbage collection: pins, finalizers, reachability sweeps, and status (spec §6.11, §9).
//!
//! GC roots are pins (and, in a consolidated registry, tags — here the tag store lives behind the
//! delegated L1/L2 layer, so this tracer roots on pins + the edge graph). A sweep marks everything
//! reachable from the roots over `owns`/`composed-of` edges (§9.2) and reports what would be evicted.
//! A **finalizer** (a pin carrying a controller) blocks unpin until released (§3.8). This is a
//! single-node tracer of the GC *semantics*; physical byte eviction is the backend store's concern.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use hologram_space::{address_bytes, KappaStore};
use uor_distribution::ErrorCode;

use crate::http_util::{json_field, parse_request, read_body, write_error, write_resp};

struct Pin {
    protected: String,
    controller: String, // empty ⇒ ordinary pin; non-empty ⇒ finalizer
    pin_kappa: String,
}

fn pins() -> &'static Mutex<HashMap<String, Vec<Pin>>> {
    static P: OnceLock<Mutex<HashMap<String, Vec<Pin>>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(HashMap::new()))
}

struct SweepStat {
    sweep_id: String,
    scanned: usize,
    reachable: usize,
    evicted: usize,
    finalizers: Vec<(String, String)>,
}

fn last_sweep() -> &'static Mutex<HashMap<String, SweepStat>> {
    static S: OnceLock<Mutex<HashMap<String, SweepStat>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

fn new_sweep_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("sweep-{}", N.fetch_add(1, Ordering::Relaxed))
}

/// `POST /v2/{path}/gc/{pin,unpin,sweep}` and `GET /v2/{path}/gc/status` (§6.11).
pub fn handle_gc(
    stream: &mut TcpStream,
    head: &[u8],
    store: &dyn KappaStore,
) -> std::io::Result<()> {
    let (method, path, _query, content_length, split) = parse_request(head);
    let Some(gpos) = path.find("/gc/") else {
        return write_error(stream, 404, "Not Found", ErrorCode::NameInvalid, "not a gc route");
    };
    let prefix = &path[4..gpos];
    let action = &path[gpos + "/gc/".len()..];

    match (method, action) {
        ("POST", "pin") => {
            let body = read_body(stream, head, split, content_length)?;
            let bs = String::from_utf8_lossy(&body);
            let Some(kappa) = json_field(&bs, "kappa") else {
                return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "no kappa");
            };
            let ttl = json_field(&bs, "ttl").unwrap_or_else(|| "0".into());
            let controller = json_field(&bs, "controller").unwrap_or_default();
            // The pin is itself a content-addressed blob (§3.8).
            let pin_content = format!("pin:{kappa}:{ttl}:{controller}");
            let pin_kappa = address_bytes(pin_content.as_bytes());
            let _ = store.put("blake3", pin_content.as_bytes());
            pins()
                .lock()
                .unwrap()
                .entry(prefix.to_string())
                .or_default()
                .push(Pin {
                    protected: kappa,
                    controller,
                    pin_kappa: pin_kappa.as_str().to_string(),
                });
            write_resp(
                stream,
                201,
                "Created",
                &[("Content-Length", "0"), ("X-Kappa-Label", pin_kappa.as_str())],
                b"",
            )
        }
        ("POST", "unpin") => {
            let body = read_body(stream, head, split, content_length)?;
            let bs = String::from_utf8_lossy(&body);
            let Some(pin_kappa) = json_field(&bs, "pin_kappa") else {
                return write_error(stream, 400, "Bad Request", ErrorCode::NameInvalid, "no pin_kappa");
            };
            let release = json_field(&bs, "release").as_deref() == Some("true");
            let mut map = pins().lock().unwrap();
            let list = map.entry(prefix.to_string()).or_default();
            let Some(idx) = list.iter().position(|p| p.pin_kappa == pin_kappa) else {
                return write_error(stream, 404, "Not Found", ErrorCode::BlobUnknown, "no such pin");
            };
            // A finalizer blocks unpin until explicitly released (§3.8, §6.11).
            if !list[idx].controller.is_empty() && !release {
                return write_error(
                    stream,
                    409,
                    "Conflict",
                    ErrorCode::FinalizerOutstanding,
                    "unpin blocked by an outstanding finalizer",
                );
            }
            list.remove(idx);
            write_resp(stream, 200, "OK", &[("Content-Length", "0")], b"")
        }
        ("POST", "sweep") => {
            let stat = compute_sweep(prefix);
            let sweep_id = stat.sweep_id.clone();
            last_sweep().lock().unwrap().insert(prefix.to_string(), stat);
            let body = format!(r#"{{"sweep_id":"{sweep_id}"}}"#);
            write_resp(
                stream,
                202,
                "Accepted",
                &[
                    ("Content-Type", "application/json"),
                    ("Content-Length", &body.len().to_string()),
                ],
                body.as_bytes(),
            )
        }
        ("GET", "status") => {
            let map = last_sweep().lock().unwrap();
            let body = match map.get(prefix) {
                Some(s) => {
                    let fin = s
                        .finalizers
                        .iter()
                        .map(|(k, c)| format!(r#"{{"kappa":"{k}","controller":"{c}"}}"#))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!(
                        r#"{{"last_sweep":"{}","objects_scanned":{},"objects_reachable":{},"objects_evicted":{},"pending_finalizers":[{fin}]}}"#,
                        s.sweep_id, s.scanned, s.reachable, s.evicted
                    )
                }
                None => r#"{"objects_scanned":0,"objects_reachable":0,"objects_evicted":0,"pending_finalizers":[]}"#.to_string(),
            };
            write_resp(
                stream,
                200,
                "OK",
                &[
                    ("Content-Type", "application/json"),
                    ("Content-Length", &body.len().to_string()),
                ],
                body.as_bytes(),
            )
        }
        _ => write_resp(
            stream,
            405,
            "Method Not Allowed",
            &[("Content-Length", "0")],
            b"",
        ),
    }
}

/// Mark-and-report sweep (§9.3): roots = pins + tags; reachable = BFS over `owns`/`composed-of` edges.
fn compute_sweep(path: &str) -> SweepStat {
    let (mut roots, finalizers): (Vec<String>, Vec<(String, String)>) = {
        let map = pins().lock().unwrap();
        let list = map.get(path).map(Vec::as_slice).unwrap_or(&[]);
        (
            list.iter().map(|p| p.protected.clone()).collect(),
            list.iter()
                .filter(|p| !p.controller.is_empty())
                .map(|p| (p.protected.clone(), p.controller.clone()))
                .collect(),
        )
    };
    // Every tagged κ is also a GC root (§9.1) — reach across the L1/L2 delegation boundary for tags.
    for (_name, kappa) in hologram_net::http::kd::tags_for(path) {
        roots.push(kappa);
    }
    let edges = crate::edge::edges_for(path);

    let mut universe: HashSet<String> = HashSet::new();
    for (s, _, t) in &edges {
        universe.insert(s.clone());
        universe.insert(t.clone());
    }
    for r in &roots {
        universe.insert(r.clone());
    }

    let mut reachable: HashSet<String> = roots.iter().cloned().collect();
    let mut queue: VecDeque<String> = roots.into_iter().collect();
    while let Some(node) = queue.pop_front() {
        for (s, rel, t) in &edges {
            if *s == node && (rel == "owns" || rel == "composed-of") && reachable.insert(t.clone()) {
                queue.push_back(t.clone());
            }
        }
    }

    let scanned = universe.len();
    let reachable_n = reachable.intersection(&universe).count();
    SweepStat {
        sweep_id: new_sweep_id(),
        scanned,
        reachable: reachable_n,
        evicted: scanned - reachable_n,
        finalizers,
    }
}
