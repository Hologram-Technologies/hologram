//! **Container lifecycle** — the generic, space-agnostic session that drives the
//! substrate's [`ContainerRuntime`] through boot → suspend to a κ snapshot →
//! resume → migrate → terminate (spec §4; arc42 chapter 5, *Boot Layer*).
//!
//! A [`Session`] holds a **container manifest κ** (the Container ID the runtime
//! spawns) and a **capability-set κ** (the authority it runs under) directly —
//! not any space-specific definition. Higher layers (e.g. holospaces) wrap it
//! with their own definition + reconfiguration; the lifecycle itself is hologram
//! infrastructure and knows nothing about them.

use core::fmt;

use hologram_space::{ContainerHandle, ContainerRuntime, KappaLabel71, RuntimeError};

/// The phase of a container's lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Defined and κ-addressed, not yet running.
    Provisioned,
    /// Spawned and running on a peer.
    Running,
    /// Halted to a κ snapshot; resumable here or on another instance.
    Suspended,
    /// Ended; not resumable.
    Terminated,
}

/// A running session of a container on one peer — driving the substrate's
/// [`ContainerRuntime`] (spec §4; arc42 chapter 5, *Boot Layer*; chapter 8,
/// *Identity and sync*).
///
/// `boot` spawns the container's manifest κ (its Container ID) under its
/// capability-set κ; `suspend` captures a snapshot κ; `resume` restarts from
/// it. Because a snapshot is content (a κ), a container suspended on one
/// instance can be resumed on another ([`Session::adopt`], migration QS2).
///
/// The session holds the container manifest κ and the capability-set κ
/// directly, so it is space-agnostic: any layer that can produce those two
/// κ-labels can drive the lifecycle through it.
pub struct Session<'r, R: ContainerRuntime> {
    runtime: &'r R,
    container: KappaLabel71,
    caps: KappaLabel71,
    handle: Option<ContainerHandle>,
    phase: Phase,
    snapshot: Option<KappaLabel71>,
}

impl<'r, R: ContainerRuntime> Session<'r, R> {
    /// Begin a session for a provisioned container, bound to a runtime.
    pub fn provision(runtime: &'r R, container: KappaLabel71, caps: KappaLabel71) -> Self {
        Self {
            runtime,
            container,
            caps,
            handle: None,
            phase: Phase::Provisioned,
            snapshot: None,
        }
    }

    /// Adopt a container suspended elsewhere from its snapshot κ (migration,
    /// QS2) — ready to [`resume`](Session::resume) on this instance.
    pub fn adopt(
        runtime: &'r R,
        container: KappaLabel71,
        caps: KappaLabel71,
        snapshot: KappaLabel71,
    ) -> Self {
        Self {
            runtime,
            container,
            caps,
            handle: None,
            phase: Phase::Suspended,
            snapshot: Some(snapshot),
        }
    }

    /// The current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The current κ snapshot, if suspended.
    pub fn snapshot(&self) -> Option<&KappaLabel71> {
        self.snapshot.as_ref()
    }

    /// The container manifest κ (the Container ID) under management.
    pub fn container(&self) -> &KappaLabel71 {
        &self.container
    }

    /// The capability-set κ the session spawns under.
    pub fn caps(&self) -> &KappaLabel71 {
        &self.caps
    }

    /// Replace the effective capability-set κ — used by a wrapping layer after a
    /// capability change (e.g. holospaces' reconfigure) so a later
    /// [`resume`](Session::resume) spawns under the new authority.
    pub fn set_caps(&mut self, caps: KappaLabel71) {
        self.caps = caps;
    }

    /// Boot the container: spawn its Container ID under its capability set.
    ///
    /// # Errors
    ///
    /// [`LifecycleError`] unless `Provisioned`, or on a runtime failure.
    pub async fn boot(&mut self) -> Result<(), LifecycleError> {
        self.expect(Phase::Provisioned, "boot")?;
        let handle = self
            .runtime
            .spawn(&self.container, &self.caps)
            .await
            .map_err(LifecycleError::Runtime)?;
        self.handle = Some(handle);
        self.phase = Phase::Running;
        Ok(())
    }

    /// Suspend the running container, capturing its state as a κ snapshot.
    ///
    /// # Errors
    ///
    /// [`LifecycleError`] unless `Running`, or on a runtime failure.
    pub async fn suspend(&mut self) -> Result<KappaLabel71, LifecycleError> {
        self.expect(Phase::Running, "suspend")?;
        let handle = self.handle.ok_or(LifecycleError::Phase {
            from: self.phase,
            action: "suspend",
        })?;
        let snapshot = self
            .runtime
            .suspend(handle)
            .await
            .map_err(LifecycleError::Runtime)?;
        self.handle = None;
        self.phase = Phase::Suspended;
        self.snapshot = Some(snapshot);
        Ok(snapshot)
    }

    /// Resume a suspended container from its snapshot κ under its capability set.
    ///
    /// # Errors
    ///
    /// [`LifecycleError`] unless `Suspended` with a snapshot, or on a runtime
    /// failure.
    pub async fn resume(&mut self) -> Result<(), LifecycleError> {
        self.expect(Phase::Suspended, "resume")?;
        let snapshot = self.snapshot.ok_or(LifecycleError::Phase {
            from: self.phase,
            action: "resume",
        })?;
        let handle = self
            .runtime
            .resume(&snapshot, &self.caps)
            .await
            .map_err(LifecycleError::Runtime)?;
        self.handle = Some(handle);
        self.phase = Phase::Running;
        Ok(())
    }

    /// Terminate the container. Allowed from any phase but `Terminated`.
    ///
    /// # Errors
    ///
    /// [`LifecycleError`] if already `Terminated`, or on a runtime failure.
    pub async fn terminate(&mut self) -> Result<(), LifecycleError> {
        if self.phase == Phase::Terminated {
            return Err(LifecycleError::Phase {
                from: self.phase,
                action: "terminate",
            });
        }
        if let Some(handle) = self.handle.take() {
            self.runtime
                .terminate(handle)
                .await
                .map_err(LifecycleError::Runtime)?;
        }
        self.phase = Phase::Terminated;
        Ok(())
    }

    fn expect(&self, phase: Phase, action: &'static str) -> Result<(), LifecycleError> {
        if self.phase == phase {
            Ok(())
        } else {
            Err(LifecycleError::Phase {
                from: self.phase,
                action,
            })
        }
    }
}

/// A failed lifecycle transition.
#[derive(Debug)]
pub enum LifecycleError {
    /// The transition is not valid from the current phase.
    Phase {
        /// The phase the container was in.
        from: Phase,
        /// The action that was attempted.
        action: &'static str,
    },
    /// The substrate runtime rejected the transition.
    Runtime(RuntimeError),
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleError::Phase { from, action } => {
                write!(f, "cannot '{action}' a container in phase {from:?}")
            }
            LifecycleError::Runtime(e) => write!(f, "substrate runtime error: {e:?}"),
        }
    }
}

impl core::error::Error for LifecycleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MockEngine, Runtime};
    use hologram_space::{Capabilities, CapabilitySet, ContainerManifest, KappaStore, Realization};
    use hologram_tck::MemKappaStore;

    fn caps() -> Capabilities {
        Capabilities {
            storage_roots: alloc::vec::Vec::new(),
            storage_quota_bytes: 0,
            network_fetch: false,
            network_announce: false,
            publish_channels: alloc::vec::Vec::new(),
            subscribe_channels: alloc::vec::Vec::new(),
            memory_max_bytes: 1 << 20,
            cpu_time_per_event_ms: 100,
            priority_weight: 0,
        }
    }

    /// Provision a container (manifest + code + caps) into a store; return
    /// (container manifest κ, capability-set κ).
    fn provision(store: &MemKappaStore, body: &[u8], state: &[u8]) -> (KappaLabel71, KappaLabel71) {
        let code = store.put("blake3", body).unwrap();
        let st = store.put("blake3", state).unwrap();
        let params = store.put("blake3", b"params").unwrap();
        let manifest = ContainerManifest {
            code,
            initial_state: st,
            parameters: params,
        };
        let cid = store.put("blake3", &manifest.canonicalize()).unwrap();
        let caps_k = store
            .put("blake3", &CapabilitySet::new(caps()).canonicalize())
            .unwrap();
        (cid, caps_k)
    }

    #[test]
    fn generic_session_drives_provision_boot_suspend_resume_terminate() {
        pollster::block_on(async {
            let store = MemKappaStore::new();
            let (cid, caps_k) = provision(&store, b"<wasm>", b"INIT");
            let rt = Runtime::new(MockEngine, store);

            let mut session = Session::provision(&rt, cid, caps_k);
            assert_eq!(session.phase(), Phase::Provisioned);
            assert_eq!(session.container(), &cid);
            assert_eq!(session.caps(), &caps_k);
            assert!(session.snapshot().is_none());

            session.boot().await.unwrap();
            assert_eq!(session.phase(), Phase::Running);

            let snap = session.suspend().await.unwrap();
            assert_eq!(session.phase(), Phase::Suspended);
            assert_eq!(session.snapshot(), Some(&snap));

            session.resume().await.unwrap();
            assert_eq!(session.phase(), Phase::Running);

            session.terminate().await.unwrap();
            assert_eq!(session.phase(), Phase::Terminated);
        });
    }

    #[test]
    fn generic_session_adopt_resumes_from_a_foreign_snapshot() {
        pollster::block_on(async {
            let store = MemKappaStore::new();
            let (cid, caps_k) = provision(&store, b"<wasm>", b"INIT");
            let rt = Runtime::new(MockEngine, store);

            // Boot + suspend one session to mint a real snapshot κ.
            let mut origin = Session::provision(&rt, cid, caps_k);
            origin.boot().await.unwrap();
            let snap = origin.suspend().await.unwrap();

            // A fresh session adopts that snapshot and resumes it (migration QS2).
            let mut adopted = Session::adopt(&rt, cid, caps_k, snap);
            assert_eq!(adopted.phase(), Phase::Suspended);
            assert_eq!(adopted.snapshot(), Some(&snap));
            adopted.resume().await.unwrap();
            assert_eq!(adopted.phase(), Phase::Running);
        });
    }

    #[test]
    fn generic_session_rejects_out_of_phase_transitions() {
        pollster::block_on(async {
            let store = MemKappaStore::new();
            let (cid, caps_k) = provision(&store, b"<wasm>", b"INIT");
            let rt = Runtime::new(MockEngine, store);

            let mut session = Session::provision(&rt, cid, caps_k);
            // Cannot resume a provisioned (never-suspended) session.
            assert!(matches!(
                session.resume().await,
                Err(LifecycleError::Phase { .. })
            ));
        });
    }

    #[test]
    fn set_caps_replaces_the_effective_capability_set() {
        pollster::block_on(async {
            let store = MemKappaStore::new();
            let (cid, caps_k) = provision(&store, b"<wasm>", b"INIT");
            let other = store.put("blake3", b"other-caps").unwrap();
            let rt = Runtime::new(MockEngine, store);

            let mut session = Session::provision(&rt, cid, caps_k);
            session.set_caps(other);
            assert_eq!(session.caps(), &other);
        });
    }
}
