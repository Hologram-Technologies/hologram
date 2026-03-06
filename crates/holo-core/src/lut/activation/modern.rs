//! Modern activation tables: GELU, SiLU (Swish).

use super::math::signed;

/// GELU: x * Phi(x). Used in BERT, GPT, ViT.
pub static GELU_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = signed(i) as f64 / 32.0;
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
        let scaled = (gelu + 4.0) * 31.875;
        t[i as usize] = if scaled < 0.0 {
            0
        } else if scaled > 255.0 {
            255
        } else {
            scaled as u8
        };
        i += 1;
    }
    t
};

/// SiLU (Swish): x * sigmoid(x). Used in EfficientNet, ConvNeXt.
pub static SILU_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = signed(i) as f64 / 16.0;
        let sigmoid = if x > 6.0 {
            1.0
        } else if x < -6.0 {
            0.0
        } else {
            let exp_neg_x = if x >= 0.0 {
                let u = 1.0 - x / 8.0;
                if u > 0.0 {
                    u * u * u * u * u * u * u * u
                } else {
                    0.0
                }
            } else {
                let u = 1.0 + (-x) / 8.0;
                u * u * u * u * u * u * u * u
            };
            1.0 / (1.0 + exp_neg_x)
        };
        let silu = x * sigmoid;
        let scaled = (silu + 1.0) * 28.333;
        t[i as usize] = if scaled < 0.0 {
            0
        } else if scaled > 255.0 {
            255
        } else {
            scaled as u8
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
        let mid = GELU_256[0];
        assert!((120..=136).contains(&mid));
        assert!(GELU_256[127] > 200);
    }

    #[test]
    fn silu_bounds() {
        assert!(SILU_256[128] < 50);
        assert!(SILU_256[127] > 200);
    }
}
