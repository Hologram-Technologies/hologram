//! `Backend` trait (spec IX.1).

use crate::error::BackendError;
use crate::kernel_call::KernelCall;
use crate::workspace::Workspace;
use prism::vocabulary::HostBounds;

pub trait Backend {
    type Bounds: HostBounds;
    type WS: Workspace;

    /// Dispatch a kernel call against the workspace.
    fn dispatch(&mut self, call: &KernelCall, workspace: &mut Self::WS)
        -> Result<(), BackendError>;
}
