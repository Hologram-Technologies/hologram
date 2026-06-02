//! Source op-name lookup.

/// Parse a canonical snake-case op name into an `OpKind`.
pub fn parse(name: &str) -> Option<hologram_graph::OpKind> {
    hologram_graph::OpKind::ALL
        .iter()
        .copied()
        .find(|kind| kind.name() == name)
}
