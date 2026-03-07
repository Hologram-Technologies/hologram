//! Precomputed lookup tables for O(1) observable and activation operations.
//!
//! All tables are computed at compile time and stored in read-only memory.
//! Lookups are O(1) with single array index operations.
//! Tables fit in L1/L2 cache for optimal latency.

pub mod activation;
pub mod arith;
pub mod q0;

// Re-export all public items for ergonomic access.
pub use activation::*;
pub use arith::*;
pub use q0::*;

/// Total static memory usage for all LUT tables (~519 KB).
pub const LUT_TOTAL_SIZE: usize = 256 * 4       // q0 unary
    + 256 * 256 * 4                              // arith
    + 256 * 256 * 2 * 2                          // GF(2) + GF(3)
    + 256 * 21                                   // activation
    + 256 * 3; // torus + orbit

/// Compose two 256-entry LUT tables: `result[i] = b[a[i]]`.
pub const fn compose_tables(a: &[u8; 256], b: &[u8; 256]) -> [u8; 256] {
    let mut result = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        result[i] = b[a[i] as usize];
        i += 1;
    }
    result
}

/// Compose a chain of activation tables at runtime.
///
/// Returns `None` if any name is unknown.
pub fn compose_chain(names: &[&str]) -> Option<[u8; 256]> {
    if names.is_empty() {
        return None;
    }
    let first = activation::activation_table(names[0])?;
    let mut result = *first;
    for name in &names[1..] {
        let table = activation::activation_table(name)?;
        result = compose_tables(&result, table);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_total_size_fits_l2() {
        const { assert!(LUT_TOTAL_SIZE < 768 * 1024) };
    }

    #[test]
    fn compose_two_tables() {
        let composed = compose_tables(&SIGMOID_256, &TANH_256);
        for i in 0..256 {
            assert_eq!(composed[i], TANH_256[SIGMOID_256[i] as usize]);
        }
    }

    #[test]
    fn compose_chain_scientific() {
        let composed = compose_chain(&["sin", "square"]).unwrap();
        for i in 0..256 {
            let step1 = SIN_256[i];
            let step2 = SQUARE_256[step1 as usize];
            assert_eq!(composed[i], step2);
        }
    }

    #[test]
    fn compose_chain_unknown() {
        assert!(compose_chain(&["sigmoid", "unknown"]).is_none());
    }
}
