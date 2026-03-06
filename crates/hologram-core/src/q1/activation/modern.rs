//! Modern Q1 activation tables: GELU, SiLU (Swish).
//!
//! Each table is `[u16; 65536]` = 128 KB, computed at compile time.

use super::math::signed16;

/// GELU for Q1: x * Phi(x). Used in BERT, GPT, ViT.
#[allow(long_running_const_eval)]
pub static GELU_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = signed16(i) as f64 / 4096.0;
        let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        let tanh_val = if inner > 4.0 {
            1.0
        } else if inner < -4.0 {
            -1.0
        } else {
            let x2 = inner * inner;
            inner * (27.0 + x2) / (27.0 + 9.0 * x2)
        };
        let gelu = 0.5 * x * (1.0 + tanh_val);
        // Map to [0, 65535]: gelu ∈ ~[-0.17, 8], normalize around 0 → 32768
        let scaled = (gelu + 1.0) * (65535.0 / 9.0);
        t[i as usize] = if scaled < 0.0 {
            0
        } else if scaled > 65535.0 {
            65535
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// SiLU (Swish) for Q1: x * sigmoid(x). Used in EfficientNet, ConvNeXt.
#[allow(long_running_const_eval)]
pub static SILU_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = signed16(i) as f64 / 4096.0;
        let sigmoid = if x > 6.0 {
            1.0
        } else if x < -6.0 {
            0.0
        } else {
            let hx = x * 0.5;
            let inner = if hx > 4.0 {
                1.0
            } else if hx < -4.0 {
                -1.0
            } else {
                let x2 = hx * hx;
                hx * (27.0 + x2) / (27.0 + 9.0 * x2)
            };
            0.5 * (1.0 + inner)
        };
        let silu = x * sigmoid;
        // Map to [0, 65535]: silu ∈ ~[-0.28, 8], normalize
        let scaled = (silu + 1.0) * (65535.0 / 9.0);
        t[i as usize] = if scaled < 0.0 {
            0
        } else if scaled > 65535.0 {
            65535
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_bounds() {
        // At x=0, GELU(0) = 0 → maps to (0+1)*65535/9 ≈ 7282
        let at_zero = GELU_65536[0];
        assert!(at_zero > 5000 && at_zero < 10000, "gelu(0) = {at_zero}");
        // Large positive → near max
        assert!(GELU_65536[32767] > 60000);
        // Large negative → GELU ≈ 0, maps near (0+1)*65535/9 ≈ 7282
        assert!(
            GELU_65536[32768] < 8000,
            "gelu(-max) = {}",
            GELU_65536[32768]
        );
    }

    #[test]
    fn silu_bounds() {
        // At x=0, SiLU(0) = 0 → maps to (0+1)*65535/9 ≈ 7282
        let at_zero = SILU_65536[0];
        assert!(at_zero > 5000 && at_zero < 10000, "silu(0) = {at_zero}");
        // Large positive → near max
        assert!(SILU_65536[32767] > 60000);
        // Large negative → SiLU ≈ 0, maps near (0+1)*65535/9 ≈ 7282
        assert!(
            SILU_65536[32768] < 8000,
            "silu(-max) = {}",
            SILU_65536[32768]
        );
    }

    #[test]
    fn gelu_near_identity_positive() {
        // For large positive x, GELU(x) ≈ x, so table should be monotonically increasing
        for i in (1u32..32000).step_by(1000) {
            assert!(
                GELU_65536[i as usize] <= GELU_65536[(i + 1) as usize],
                "gelu not monotonic at {i}"
            );
        }
    }

    #[test]
    fn silu_near_identity_positive() {
        // For large positive x, SiLU(x) ≈ x, so monotonic
        for i in (1u32..32000).step_by(1000) {
            assert!(
                SILU_65536[i as usize] <= SILU_65536[(i + 1) as usize],
                "silu not monotonic at {i}"
            );
        }
    }
}
