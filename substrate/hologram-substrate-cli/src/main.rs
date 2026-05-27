//! The `hologram` node binary (spec §9.2). Thin shell: parse args + read files, then delegate to
//! `hologram_substrate_cli::run` against a native redb store. Container/network verbs
//! (`spawn`/`serve`) arrive with the Wasmtime engine / libp2p transport backends.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use hologram_store_native::NativeKappaStore;
use hologram_substrate_cli::{parse_kappa, run, CliError, Command, Outcome};

#[derive(Parser)]
#[command(name = "hologram", about = "Hologram deployment-substrate node (κ-label store/route)")]
struct Cli {
    /// Path to the node's redb store.
    #[arg(long, default_value = "hologram-store.redb")]
    store: PathBuf,
    #[command(subcommand)]
    verb: Verb,
}

#[derive(Subcommand)]
enum Verb {
    /// Put a file's bytes under a σ-axis; print the κ-label.
    Put {
        #[arg(long, default_value = "blake3")]
        axis: String,
        file: PathBuf,
    },
    /// Write a κ-label's canonical bytes to stdout.
    Get { kappa: String },
    /// Pin a κ-label as a reachability root.
    Pin { kappa: String },
    /// Remove a pin.
    Unpin { kappa: String },
    /// Walk reachability from pinned roots; evict unreachable bytes. Prints the eviction count.
    Gc,
    /// List locally-present κ-labels.
    Ls,
    /// Show a stored artifact's realization IRI and its embedded references (SPINE-2/3).
    Inspect { kappa: String },
    /// Re-derive a file's bytes through the σ-axis and check they match a κ-label (SPINE-4).
    Verify { kappa: String, file: PathBuf },
    /// Build a Container manifest from three κ-labels (code, state, params); print the Container ID.
    Manifest { code: String, state: String, params: String },
    /// Spawn a real Wasm container (Wasmtime), optionally deliver one event file, then suspend;
    /// print the snapshot κ-label.
    Spawn {
        /// Container ID κ-label.
        container: String,
        /// Capability Set κ-label.
        caps: String,
        /// Optional event payload file delivered once before suspend.
        #[arg(long)]
        event: Option<PathBuf>,
    },
    /// Run the HTTP-CAS gateway over this node's store (spec §6.5). Blocks until terminated.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
    },
    /// Mint a Capability Set κ-label (grants + budgets). Empty by default.
    Caps {
        /// Readable storage-root κ-labels (repeatable).
        #[arg(long = "root")]
        roots: Vec<String>,
        /// Publishable channel κ-labels (repeatable).
        #[arg(long = "publish")]
        publish: Vec<String>,
        /// Subscribable channel κ-labels (repeatable).
        #[arg(long = "subscribe")]
        subscribe: Vec<String>,
        #[arg(long, default_value_t = 0)]
        quota: u64,
        #[arg(long = "mem", default_value_t = 0)]
        memory_max: u64,
        #[arg(long = "cpu-ms", default_value_t = 0)]
        cpu_ms: u64,
        #[arg(long)]
        fetch: bool,
        #[arg(long)]
        announce: bool,
    },
}

fn build(verb: Verb) -> Result<Command, String> {
    let rd = |p: &PathBuf| std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()));
    Ok(match verb {
        Verb::Put { axis, file } => Command::Put { axis, bytes: rd(&file)? },
        Verb::Get { kappa } => Command::Get(parse_kappa(&kappa).map_err(badk)?),
        Verb::Pin { kappa } => Command::Pin(parse_kappa(&kappa).map_err(badk)?),
        Verb::Unpin { kappa } => Command::Unpin(parse_kappa(&kappa).map_err(badk)?),
        Verb::Gc => Command::Gc,
        Verb::Ls => Command::Ls,
        Verb::Inspect { kappa } => Command::Inspect(parse_kappa(&kappa).map_err(badk)?),
        Verb::Verify { kappa, file } => {
            Command::Verify { kappa: parse_kappa(&kappa).map_err(badk)?, bytes: rd(&file)? }
        }
        Verb::Manifest { code, state, params } => Command::Manifest {
            code: parse_kappa(&code).map_err(badk)?,
            initial_state: parse_kappa(&state).map_err(badk)?,
            parameters: parse_kappa(&params).map_err(badk)?,
        },
        Verb::Caps { roots, publish, subscribe, quota, memory_max, cpu_ms, fetch, announce } => {
            let ks = |v: Vec<String>| v.iter().map(|s| parse_kappa(s)).collect::<Result<Vec<_>, _>>().map_err(badk);
            Command::Caps(hologram_substrate_core::Capabilities {
                storage_roots: ks(roots)?,
                publish_channels: ks(publish)?,
                subscribe_channels: ks(subscribe)?,
                storage_quota_bytes: quota,
                memory_max_bytes: memory_max,
                cpu_time_per_event_ms: cpu_ms,
                network_fetch: fetch,
                network_announce: announce,
            })
        }
        Verb::Spawn { .. } | Verb::Serve { .. } => unreachable!("handled in main()"),
    })
}

fn badk(_: CliError) -> String {
    "malformed κ-label (expected <axis>:<hex>)".into()
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.verb {
        // Container/network verbs need the runtime/engine/server, not the store-generic `run`.
        Verb::Spawn { container, caps, event } => run_spawn(&cli.store, &container, &caps, event),
        Verb::Serve { listen } => run_serve(&cli.store, &listen),
        // Storage/addressing verbs.
        other => {
            let store = match NativeKappaStore::open(&cli.store) {
                Ok(s) => s,
                Err(e) => return fail(format!("open store: {e:?}")),
            };
            let cmd = match build(other) {
                Ok(c) => c,
                Err(e) => return fail(e),
            };
            match run(&store, cmd) {
                Ok(out) => render(out),
                Err(e) => fail(format!("{e:?}")),
            }
        }
    }
}

/// `hologram spawn` — run a real Wasm container, deliver an optional event, suspend, print snapshot κ.
fn run_spawn(store_path: &std::path::Path, container: &str, caps: &str, event: Option<PathBuf>) -> ExitCode {
    use hologram_runtime::Runtime;
    use hologram_runtime_wasmtime::WasmtimeEngine;
    use hologram_substrate_core::ContainerRuntime;

    let cid = match parse_kappa(container) {
        Ok(k) => k,
        Err(_) => return fail("malformed container-id κ-label".into()),
    };
    let ck = match parse_kappa(caps) {
        Ok(k) => k,
        Err(_) => return fail("malformed capabilities κ-label".into()),
    };
    let store = match NativeKappaStore::open(store_path) {
        Ok(s) => s,
        Err(e) => return fail(format!("open store: {e:?}")),
    };
    let rt = Runtime::new(WasmtimeEngine::new(), store);
    pollster::block_on(async {
        let h = match rt.spawn(&cid, &ck).await {
            Ok(h) => h,
            Err(e) => return fail(format!("spawn: {e:?}")),
        };
        if let Some(ev) = event {
            match std::fs::read(&ev) {
                Ok(bytes) => {
                    let _ = rt.deliver_event(h, &bytes);
                }
                Err(e) => return fail(format!("read event {}: {e}", ev.display())),
            }
        }
        match rt.suspend(h).await {
            Ok(snap) => {
                println!("{}", snap.as_str());
                ExitCode::SUCCESS
            }
            Err(e) => fail(format!("suspend: {e:?}")),
        }
    })
}

/// `hologram serve` — run the HTTP-CAS gateway over this node's store; block until terminated.
fn run_serve(store_path: &std::path::Path, listen: &str) -> ExitCode {
    let store = match NativeKappaStore::open(store_path) {
        Ok(s) => std::sync::Arc::new(s),
        Err(e) => return fail(format!("open store: {e:?}")),
    };
    let server = match hologram_net_http::live::serve_addr(store, listen, false) {
        Ok(s) => s,
        Err(e) => return fail(format!("listen on {listen}: {e}")),
    };
    eprintln!("hologram: HTTP-CAS gateway on http://{}/cas/{{kappa}}", server.addr());
    // Park; the server thread handles requests until the process is terminated.
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn render(out: Outcome) -> ExitCode {
    match out {
        Outcome::Kappa(k) => println!("{}", k.as_str()),
        Outcome::Data(d) => {
            let _ = std::io::stdout().write_all(&d);
        }
        Outcome::Labels(ks) => {
            for k in ks {
                println!("{}", k.as_str());
            }
        }
        Outcome::Inspected { iri, refs } => {
            println!("realization: {iri}");
            for r in refs {
                println!("  ref: {}", r.as_str());
            }
        }
        Outcome::Count(n) => println!("{n}"),
        Outcome::Verified(ok) => {
            println!("{ok}");
            if !ok {
                return ExitCode::from(2);
            }
        }
        Outcome::Pinned => println!("pinned"),
        Outcome::Unpinned => println!("unpinned"),
    }
    ExitCode::SUCCESS
}

fn fail(msg: String) -> ExitCode {
    eprintln!("hologram: {msg}");
    ExitCode::FAILURE
}
