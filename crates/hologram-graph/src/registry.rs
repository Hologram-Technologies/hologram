//! Shape / dtype registries (spec VI.4).

use alloc::boxed::Box;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DTypeId(pub u8);

/// Concrete shape descriptor. Small-rank fast path inline; rank > 8
/// overflows to heap.
#[derive(Debug, Clone)]
pub struct ShapeDescriptor {
    pub rank: u8,
    pub dims: [u64; 8],
    pub dims_overflow: Option<Box<[u64]>>,
}

impl ShapeDescriptor {
    pub fn rank1(d0: u64) -> Self {
        let mut dims = [0u64; 8];
        dims[0] = d0;
        Self {
            rank: 1,
            dims,
            dims_overflow: None,
        }
    }

    pub fn rank2(d0: u64, d1: u64) -> Self {
        let mut dims = [0u64; 8];
        dims[0] = d0;
        dims[1] = d1;
        Self {
            rank: 2,
            dims,
            dims_overflow: None,
        }
    }

    pub fn rank4(d0: u64, d1: u64, d2: u64, d3: u64) -> Self {
        let mut dims = [0u64; 8];
        dims[0] = d0;
        dims[1] = d1;
        dims[2] = d2;
        dims[3] = d3;
        Self {
            rank: 4,
            dims,
            dims_overflow: None,
        }
    }

    pub fn dim(&self, i: usize) -> Option<u64> {
        if i < self.rank as usize {
            if i < 8 {
                Some(self.dims[i])
            } else {
                self.dims_overflow
                    .as_ref()
                    .and_then(|d| d.get(i - 8).copied())
            }
        } else {
            None
        }
    }

    /// Total element count (product of dims).
    pub fn total_elements(&self) -> u64 {
        let mut p = 1u64;
        let r = self.rank as usize;
        for i in 0..r.min(8) {
            p = p.saturating_mul(self.dims[i]);
        }
        if let Some(overflow) = &self.dims_overflow {
            for &d in overflow.iter() {
                p = p.saturating_mul(d);
            }
        }
        p
    }
}

#[derive(Debug, Default, Clone)]
pub struct ShapeRegistry {
    shapes: Vec<ShapeDescriptor>,
}

impl ShapeRegistry {
    pub fn new() -> Self {
        Self { shapes: Vec::new() }
    }

    pub fn intern(&mut self, descriptor: ShapeDescriptor) -> ShapeId {
        let id = ShapeId(self.shapes.len() as u32);
        self.shapes.push(descriptor);
        id
    }

    pub fn get(&self, id: ShapeId) -> Option<&ShapeDescriptor> {
        self.shapes.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.shapes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }
}
