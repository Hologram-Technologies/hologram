//! κ-Distribution conformance levels (spec §13.1). Each level is a strict superset of the prior one:
//! a registry conforming at level N conforms at every level below N.

/// A κ-Distribution conformance level (spec §13.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConformanceLevel {
    /// Level 1 — blob operations: put/get/exists/remove/list, chunked upload with recovery,
    /// upload cancellation, mount.
    Blobs = 1,
    /// Level 2 — Level 1 + tags: tag_set/get/list/delete/set_if.
    Tags = 2,
    /// Level 3 — Level 2 + edges: edge_put/get/remove (edges as blobs).
    Edges = 3,
    /// Level 4 — Level 3 + composition, witnesses, schemas.
    Composition = 4,
    /// Level 5 — Level 4 + GC, admission, federation, multi-hop.
    Federation = 5,
}

impl ConformanceLevel {
    /// The level number (1–5).
    pub const fn number(self) -> u8 {
        self as u8
    }

    /// Whether conforming at `self` implies conforming at `other` (levels are cumulative).
    pub const fn includes(self, other: ConformanceLevel) -> bool {
        (self as u8) >= (other as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_cumulative_and_numbered() {
        assert_eq!(ConformanceLevel::Blobs.number(), 1);
        assert_eq!(ConformanceLevel::Federation.number(), 5);
        // Higher levels include lower ones; the converse does not hold.
        assert!(ConformanceLevel::Federation.includes(ConformanceLevel::Blobs));
        assert!(ConformanceLevel::Edges.includes(ConformanceLevel::Tags));
        assert!(!ConformanceLevel::Blobs.includes(ConformanceLevel::Tags));
    }
}
