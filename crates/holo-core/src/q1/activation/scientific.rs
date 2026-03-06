//! Scientific Q1 activation tables: trig, inverse trig, logarithms, exponentials, powers.
//!
//! Each table is `[u16; 65536]` = 128 KB, computed at compile time.

use crate::lut::activation::math::*;

/// Sin for Q1: input mapped as angle in [0, 2pi), output in [0, 65535] where 32768 = 0.
#[allow(long_running_const_eval)]
pub static SIN_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 65536 {
        let angle = i as f64 * TAU / 65536.0;
        let v = const_sin(angle) * 32767.0 + 32768.0;
        t[i as usize] = if v > 65535.0 {
            65535
        } else if v < 0.0 {
            0
        } else {
            v as u16
        };
        i += 1;
    }
    t
};

/// Cos for Q1: input mapped as angle in [0, 2pi), output in [0, 65535] where 32768 = 0.
#[allow(long_running_const_eval)]
pub static COS_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 65536 {
        let angle = i as f64 * TAU / 65536.0;
        let v = const_cos(angle) * 32767.0 + 32768.0;
        t[i as usize] = if v > 65535.0 {
            65535
        } else if v < 0.0 {
            0
        } else {
            v as u16
        };
        i += 1;
    }
    t
};

/// Tan for Q1: input as angle, output clamped to [-1,1] mapped to [0, 65535].
#[allow(long_running_const_eval)]
pub static TAN_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 65536 {
        let angle = i as f64 * TAU / 65536.0;
        let s = const_sin(angle);
        let c = const_cos(angle);
        let v = if c > 0.001 || c < -0.001 {
            s / c
        } else if s >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let scaled = clamped * 32767.0 + 32768.0;
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

/// Asin for Q1: input as [-1,1] mapped from [0, 65535], output as angle.
#[allow(long_running_const_eval)]
pub static ASIN_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 65536 {
        let v = (i as f64 - 32768.0) / 32767.0;
        let clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let angle = const_asin(clamped);
        let scaled = ((angle + FRAC_PI_2) / PI) * 65535.0;
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

/// Acos for Q1: input as [-1,1], output as angle.
#[allow(long_running_const_eval)]
pub static ACOS_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 65536 {
        let v = (i as f64 - 32768.0) / 32767.0;
        let clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let angle = FRAC_PI_2 - const_asin(clamped);
        let scaled = (angle / PI) * 65535.0;
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

/// Atan for Q1: input as [-1,1], output as angle.
#[allow(long_running_const_eval)]
pub static ATAN_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 65536 {
        let v = (i as f64 - 32768.0) / 32767.0;
        let angle = const_atan(v);
        let scaled = ((angle + FRAC_PI_2) / PI) * 65535.0;
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

/// Log2 for Q1: maps [1, 65535] to [0, 65535]. log2(0) = 0.
#[allow(long_running_const_eval)]
pub static LOG2_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    t[0] = 0;
    let max_log2 = const_log2(65535.0);
    let mut i = 1u32;
    while i < 65536 {
        let v = const_log2(i as f64) / max_log2;
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

/// Log10 for Q1: maps [1, 65535] to [0, 65535]. log10(0) = 0.
#[allow(long_running_const_eval)]
pub static LOG10_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    t[0] = 0;
    let log2_10 = const_log2(10.0);
    let max_log10 = const_log2(65535.0) / log2_10;
    let mut i = 1u32;
    while i < 65536 {
        let v = (const_log2(i as f64) / log2_10) / max_log10;
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

/// Exp2 for Q1: maps [0, 65535] to [0, 65535].
#[allow(long_running_const_eval)]
pub static EXP2_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = i as f64 / 65535.0 * 16.0; // map to [0, 16]
        let v = const_exp2(x) / 65536.0; // normalize by 2^16
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

/// Exp10 for Q1: maps [0, 65535] to [0, 65535].
#[allow(long_running_const_eval)]
pub static EXP10_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let log2_10 = const_log2(10.0);
    let mut i = 0u32;
    while i < 65536 {
        let x = i as f64 / 65535.0 * 4.8; // map to [0, 4.8]
        let v = const_exp2(x * log2_10) / 65535.0;
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

/// Square for Q1: maps [0, 65535] to [0, 65535] as x^2.
#[allow(long_running_const_eval)]
pub static SQUARE_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = i as f64 / 65535.0;
        let scaled = x * x * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
            65535
        } else {
            scaled as u16
        };
        i += 1;
    }
    t
};

/// Cube for Q1: maps [0, 65535] to [0, 65535] as x^3.
#[allow(long_running_const_eval)]
pub static CUBE_65536: [u16; 65536] = {
    let mut t = [0u16; 65536];
    let mut i = 0u32;
    while i < 65536 {
        let x = i as f64 / 65535.0;
        let scaled = x * x * x * 65535.0;
        t[i as usize] = if scaled > 65535.0 {
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
    fn sin_bounds() {
        // At i=0, angle=0, sin(0)=0 → midpoint ~32768
        let at_zero = SIN_65536[0];
        assert!(
            (32000..=33500).contains(&at_zero),
            "sin(0) = {at_zero}"
        );
        // At i=16384, angle=pi/2, sin(pi/2)=1 → near max
        assert!(SIN_65536[16384] > 64000);
        // At i=49152, angle=3pi/2, sin(3pi/2)=-1 → near min
        assert!(SIN_65536[49152] < 1500);
    }

    #[test]
    fn cos_bounds() {
        // At i=0, angle=0, cos(0)=1 → near max
        assert!(COS_65536[0] > 64000);
        // At i=16384, angle=pi/2, cos(pi/2)=0 → midpoint
        let at_quarter = COS_65536[16384];
        assert!(
            (32000..=33500).contains(&at_quarter),
            "cos(pi/2) = {at_quarter}"
        );
        // At i=32768, angle=pi, cos(pi)=-1 → near min
        assert!(COS_65536[32768] < 1500);
    }

    #[test]
    fn exp2_monotonic() {
        for i in (0..65535u32).step_by(256) {
            assert!(
                EXP2_65536[i as usize] <= EXP2_65536[(i + 1) as usize],
                "exp2 not monotonic at {i}"
            );
        }
    }

    #[test]
    fn square_endpoints() {
        assert_eq!(SQUARE_65536[0], 0);
        assert_eq!(SQUARE_65536[65535], 65535);
    }

    #[test]
    fn cube_endpoints() {
        assert_eq!(CUBE_65536[0], 0);
        assert_eq!(CUBE_65536[65535], 65535);
    }

    #[test]
    fn square_monotonic() {
        for i in (0..65535u32).step_by(256) {
            assert!(
                SQUARE_65536[i as usize] <= SQUARE_65536[(i + 1) as usize],
                "square not monotonic at {i}"
            );
        }
    }

    #[test]
    fn log2_monotonic() {
        for i in (1..65535u32).step_by(256) {
            assert!(
                LOG2_65536[i as usize] <= LOG2_65536[(i + 1) as usize],
                "log2 not monotonic at {i}"
            );
        }
    }

    #[test]
    fn log10_monotonic() {
        for i in (1..65535u32).step_by(256) {
            assert!(
                LOG10_65536[i as usize] <= LOG10_65536[(i + 1) as usize],
                "log10 not monotonic at {i}"
            );
        }
    }

    #[test]
    fn asin_at_zero() {
        // At midpoint (32768), input=0, asin(0)=0, maps to pi/2 normalized → ~32768
        let at_mid = ASIN_65536[32768];
        assert!(
            (32000..=33500).contains(&at_mid),
            "asin(0) = {at_mid}"
        );
    }

    #[test]
    fn acos_at_zero() {
        // At midpoint (32768), input=0, acos(0)=pi/2, maps to 0.5 → ~32768
        let at_mid = ACOS_65536[32768];
        assert!(
            (32000..=33500).contains(&at_mid),
            "acos(0) = {at_mid}"
        );
    }
}
