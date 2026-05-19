//! `Backend` trait (spec IX.1).

use prism::vocabulary::HostBounds;
use crate::kernel_call::KernelCall;
use crate::workspace::Workspace;
use crate::error::BackendError;

pub trait Backend {
    type Bounds: HostBounds;
    type WS: Workspace;

    /// Dispatch a kernel call against the workspace.
    fn dispatch(
        &mut self,
        call: &KernelCall,
        workspace: &mut Self::WS,
    ) -> Result<(), BackendError>;
}
