//! Basic Q1 activation tables: sigmoid, tanh, exp, log, relu, sqrt, abs.
//!
//! Each table is `[u16; 65536]` = 128 KB, computed at compile time.

use super::math::signed16;
use crate::lut::activation::math::*;

/// Sigmoid for Q1: piecewise linear approximation mapping signed input to [0, 65535].
#[allow(long_running_const_eval)]
pub static SIGMOID_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = signed16(i) as f64 / 4096.0; // scale to ~[-8, 8]
                                             // Const-compatible sigmoid approximation
        let sig = if x > 6.0 {
            1.0
        } else if x < -6.0 {
            0.0
        } else {
            // tanh-based: sigmoid(x) = 0.5 * (1 + tanh(x/2))
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
        let scaled = sig * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else if scaled < 0.0 {
            0
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// Tanh for Q1: maps signed input to [0, 65535] where 32768 = 0.0.
#[allow(long_running_const_eval)]
pub static TANH_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = signed16(i) as f64 / 4096.0;
        let tanh_val = if x > 4.0 {
            1.0
        } else if x < -4.0 {
            -1.0
        } else {
            let x2 = x * x;
            x * (27.0 + x2) / (27.0 + 9.0 * x2)
        };
        let scaled = (tanh_val * 0.5 + 0.5) * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else if scaled < 0.0 {
            0
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// Exp for Q1: maps signed input to [0, 65535].
#[allow(long_running_const_eval)]
pub static EXP_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = signed16(i) as f64 / 4096.0;
        let v = const_exp2(x / core::f64::consts::LN_2);
        // Normalize: map [0, exp(8)] → [0, 65535]
        let max_val = 2980.957987; // exp(8)
        let scaled = (v / max_val) * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else if scaled < 0.0 {
            0
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// Log for Q1: maps [1, 65535] to [0, 65535]. log(0) = 0.
#[allow(long_running_const_eval)]
pub static LOG_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    t[0] = 0;
    let max_log = const_log2(65535.0);
    let mut i = 1u32;
    while i < 65536 {
        let v = const_log2(i as f64) / max_log;
        let scaled = v * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else if scaled < 0.0 {
            0
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// ReLU for Q1: identity for 0..=32767 (positive), 0 for 32768..=65535 (negative).
#[allow(long_running_const_eval)]
pub static RELU_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        t[i as usize] = if i < 32768 { i as u16 } else { 0 };
        i += 1;
    }
    t
};

/// Sqrt for Q1: integer square root scaled to [0, 65535].
#[allow(long_running_const_eval)]
pub static SQRT_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let max_sqrt = const_sqrt_f64(65535.0);
    let mut i = 0u32;
    while i < 65536 {
        let v = const_sqrt_f64(i as f64) / max_sqrt;
        let scaled = v * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// Abs for Q1: |signed(x)| mapped to unsigned.
#[allow(long_running_const_eval)]
pub static ABS_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        t[i as usize] = if i < 32768 {
            i as u16
        } else {
            (65536 - i) as u16
        };
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        // Index 0 → x=0 → sigmoid(0) = 0.5 → ~32768
        let mid = SIGMOID_65536[0];
        assert!((32000..=33500).contains(&mid), "sigmoid(0) = {mid}");
        // Large positive → near max
        assert!(SIGMOID_65536[32767] > 60000);
        // Large negative → near min
        assert!(SIGMOID_65536[32768] < 5000);
    }

    #[test]
    fn sigmoid_monotonic_sampled() {
        for i in (0..65535u32).step_by(256) {
            let a = SIGMOID_65536[i as usize];
            let b = SIGMOID_65536[(i + 1) as usize];
            // Monotonic in the positive direction (signed order)
            if i < 32767 {
                assert!(a <= b, "sigmoid not monotonic at {i}: {a} > {b}");
            }
        }
    }

    #[test]
    fn tanh_bounds() {
        // Index 0 → x=0 → tanh(0) = 0 → midpoint ~32768
        let mid = TANH_65536[0];
        assert!((32000..=33500).contains(&mid), "tanh(0) = {mid}");
        assert!(TANH_65536[32767] > 60000);
        assert!(TANH_65536[32768] < 5000);
    }

    #[test]
    fn relu_values() {
        // Positive side: identity
        for i in 0..32768u16 {
            assert_eq!(RELU_65536[i as usize], i);
        }
        // Negative side: zero
        for i in (32768u32..=65535).step_by(256) {
            assert_eq!(RELU_65536[i as usize], 0);
        }
    }

    #[test]
    fn abs_values() {
        assert_eq!(ABS_65536[0], 0);
        assert_eq!(ABS_65536[1], 1);
        assert_eq!(ABS_65536[32767], 32767);
        assert_eq!(ABS_65536[65535], 1); // |-1| = 1
        assert_eq!(ABS_65536[32768], 32768); // |-32768| = 32768
    }

    #[test]
    fn log_monotonic() {
        for i in (1..65535u32).step_by(256) {
            assert!(
                LOG_65536[i as usize] <= LOG_65536[(i + 1) as usize],
                "log not monotonic at {i}"
            );
        }
    }

    #[test]
    fn sqrt_monotonic() {
        for i in (0..65535u32).step_by(256) {
            assert!(
                SQRT_65536[i as usize] <= SQRT_65536[(i + 1) as usize],
                "sqrt not monotonic at {i}"
            );
        }
    }

    #[test]
    fn sqrt_endpoints() {
        assert_eq!(SQRT_65536[0], 0);
        assert_eq!(SQRT_65536[65535], 65535);
    }

    #[test]
    fn exp_positive_large() {
        // Large positive input → near max
        assert!(EXP_65536[32767] > 60000);
        // Zero input → exp(0) = 1, relatively small in normalized scale
        let at_zero = EXP_65536[0];
        assert!(at_zero > 0 && at_zero < 32768, "exp(0) = {at_zero}");
    }
}
