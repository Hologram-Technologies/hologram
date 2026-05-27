//! `hologram` CLI command logic (spec §9.2), **store-generic and hermetically testable**: the
//! `main` shell parses args and reads files, then calls [`run`] against a `NativeKappaStore`; tests
//! call [`run`] against the in-memory reference. Every verb is a κ-label operation (SPINE-1).

use hologram_realizations::{CapabilitySet, ContainerManifest, REGISTRY};
use hologram_substrate_core::{
    references, verify_kappa, Capabilities, GarbageCollect, KappaLabel, KappaLabel71, KappaStore,
    Realization, StoreError,
};

/// A parsed CLI command (file paths already resolved to bytes / κ-labels by the shell).
pub enum Command {
    Put {
        axis: String,
        bytes: Vec<u8>,
    },
    Get(KappaLabel71),
    Pin(KappaLabel71),
    Unpin(KappaLabel71),
    Gc,
    Ls,
    Inspect(KappaLabel71),
    Verify {
        kappa: KappaLabel71,
        bytes: Vec<u8>,
    },
    Manifest {
        code: KappaLabel71,
        initial_state: KappaLabel71,
        parameters: KappaLabel71,
    },
    /// Mint a Capability Set κ-label from grants/budgets.
    Caps(Capabilities),
}

/// The result of a command, ready to render.
pub enum Outcome {
    Kappa(KappaLabel71),
    Data(Vec<u8>),
    Labels(Vec<KappaLabel71>),
    Inspected {
        iri: String,
        refs: Vec<KappaLabel71>,
    },
    Count(usize),
    Verified(bool),
    Pinned,
    Unpinned,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    Store(StoreError),
    NotFoundLocally,
    BadKappa,
    NotARealization,
    AxisUnsupported,
}

/// Parse a `<axis>:<hex>` κ-label string (blake3 width = 71 bytes).
pub fn parse_kappa(s: &str) -> Result<KappaLabel71, CliError> {
    let arr: [u8; 71] = s.as_bytes().try_into().map_err(|_| CliError::BadKappa)?;
    KappaLabel::from_bytes(&arr).map_err(|_| CliError::BadKappa)
}

fn read_iri(bytes: &[u8]) -> Result<String, CliError> {
    let nul = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or(CliError::NotARealization)?;
    core::str::from_utf8(&bytes[..nul])
        .map(|s| s.to_string())
        .map_err(|_| CliError::NotARealization)
}

/// Execute a command against any store that also supports GC. The single dispatch point shared by
/// the binary and the conformance tests.
pub fn run<S: KappaStore + GarbageCollect>(store: &S, cmd: Command) -> Result<Outcome, CliError> {
    match cmd {
        Command::Put { axis, bytes } => store
            .put(&axis, &bytes)
            .map(Outcome::Kappa)
            .map_err(CliError::Store),
        Command::Get(k) => store
            .get(&k)
            .map_err(CliError::Store)?
            .map(|b| Outcome::Data(b.to_vec()))
            .ok_or(CliError::NotFoundLocally),
        Command::Pin(k) => {
            store.pin(&k).map_err(CliError::Store)?;
            Ok(Outcome::Pinned)
        }
        Command::Unpin(k) => {
            store.unpin(&k).map_err(CliError::Store)?;
            Ok(Outcome::Unpinned)
        }
        Command::Gc => store
            .gc(REGISTRY)
            .map(Outcome::Count)
            .map_err(CliError::Store),
        Command::Ls => Ok(Outcome::Labels(store.iterate())),
        Command::Inspect(k) => {
            let bytes = store
                .get(&k)
                .map_err(CliError::Store)?
                .ok_or(CliError::NotFoundLocally)?;
            let iri = read_iri(&bytes)?;
            let refs = references(&bytes, REGISTRY).map_err(|_| CliError::NotARealization)?;
            Ok(Outcome::Inspected { iri, refs })
        }
        Command::Verify { kappa, bytes } => verify_kappa(&bytes, &kappa)
            .map(Outcome::Verified)
            .map_err(|_| CliError::BadKappa),
        Command::Manifest {
            code,
            initial_state,
            parameters,
        } => {
            let m = ContainerManifest {
                code,
                initial_state,
                parameters,
            };
            store
                .put("blake3", &m.canonicalize())
                .map(Outcome::Kappa)
                .map_err(CliError::Store)
        }
        Command::Caps(c) => store
            .put("blake3", &CapabilitySet::new(c).canonicalize())
            .map(Outcome::Kappa)
            .map_err(CliError::Store),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_store_mem::MemKappaStore;

    #[test]
    fn put_get_roundtrip_and_ls() {
        let s = MemKappaStore::new();
        let Outcome::Kappa(k) = run(
            &s,
            Command::Put {
                axis: "blake3".into(),
                bytes: b"hi".to_vec(),
            },
        )
        .unwrap() else {
            panic!()
        };
        let Outcome::Data(d) = run(&s, Command::Get(k)).unwrap() else {
            panic!()
        };
        assert_eq!(d, b"hi");
        let Outcome::Labels(ls) = run(&s, Command::Ls).unwrap() else {
            panic!()
        };
        assert_eq!(ls, vec![k]);
    }

    #[test]
    fn manifest_then_inspect_shows_iri_and_references() {
        let s = MemKappaStore::new();
        let put = |b: &[u8]| match run(
            &s,
            Command::Put {
                axis: "blake3".into(),
                bytes: b.to_vec(),
            },
        )
        .unwrap()
        {
            Outcome::Kappa(k) => k,
            _ => unreachable!(),
        };
        let (code, state, params) = (put(b"code"), put(b"state"), put(b"params"));
        let Outcome::Kappa(cid) = run(
            &s,
            Command::Manifest {
                code,
                initial_state: state,
                parameters: params,
            },
        )
        .unwrap() else {
            panic!()
        };
        let Outcome::Inspected { iri, refs } = run(&s, Command::Inspect(cid)).unwrap() else {
            panic!()
        };
        assert_eq!(
            iri,
            "https://hologram.foundation/realization/container-manifest"
        );
        assert_eq!(refs, vec![code, state, params]);
    }

    #[test]
    fn pin_then_gc_keeps_reachable() {
        let s = MemKappaStore::new();
        let put = |b: &[u8]| match run(
            &s,
            Command::Put {
                axis: "blake3".into(),
                bytes: b.to_vec(),
            },
        )
        .unwrap()
        {
            Outcome::Kappa(k) => k,
            _ => unreachable!(),
        };
        let (code, state, params) = (put(b"c"), put(b"s"), put(b"p"));
        let Outcome::Kappa(cid) = run(
            &s,
            Command::Manifest {
                code,
                initial_state: state,
                parameters: params,
            },
        )
        .unwrap() else {
            panic!()
        };
        let _orphan = put(b"orphan");
        run(&s, Command::Pin(cid)).unwrap();
        let Outcome::Count(evicted) = run(&s, Command::Gc).unwrap() else {
            panic!()
        };
        assert_eq!(evicted, 1); // the orphan only
        assert!(matches!(run(&s, Command::Get(cid)), Ok(Outcome::Data(_))));
    }

    #[test]
    fn verify_detects_tampering() {
        let s = MemKappaStore::new();
        let Outcome::Kappa(k) = run(
            &s,
            Command::Put {
                axis: "blake3".into(),
                bytes: b"authentic".to_vec(),
            },
        )
        .unwrap() else {
            panic!()
        };
        assert!(matches!(
            run(
                &s,
                Command::Verify {
                    kappa: k,
                    bytes: b"authentic".to_vec()
                }
            ),
            Ok(Outcome::Verified(true))
        ));
        assert!(matches!(
            run(
                &s,
                Command::Verify {
                    kappa: k,
                    bytes: b"forged".to_vec()
                }
            ),
            Ok(Outcome::Verified(false))
        ));
    }

    #[test]
    fn get_absent_is_not_found() {
        let s = MemKappaStore::new();
        let k = hologram_substrate_core::address_bytes(b"absent");
        assert_eq!(
            run(&s, Command::Get(k)).err(),
            Some(CliError::NotFoundLocally)
        );
    }
}
