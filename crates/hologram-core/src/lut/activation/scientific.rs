//! Scientific function tables: trig, inverse trig, logarithms, exponentials, powers.

use super::math::*;

pub static SIN_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 256 {
        let angle = i as f64 * TAU / 256.0;
        t[i as usize] = (const_sin(angle) * 127.0 + 128.0) as u8;
        i += 1;
    }
    t
};

pub static COS_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 256 {
        let angle = i as f64 * TAU / 256.0;
        t[i as usize] = (const_cos(angle) * 127.0 + 128.0) as u8;
        i += 1;
    }
    t
};

pub static TAN_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const TAU: f64 = core::f64::consts::TAU;
    while i < 256 {
        let angle = i as f64 * TAU / 256.0;
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
        t[i as usize] = (clamped * 127.0 + 128.0) as u8;
        i += 1;
    }
    t
};

pub static ASIN_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 256 {
        let v = (i as f64 - 128.0) / 127.0;
        let clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let angle = const_asin(clamped);
        t[i as usize] = (((angle + FRAC_PI_2) / PI) * 255.0) as u8;
        i += 1;
    }
    t
};

pub static ACOS_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 256 {
        let v = (i as f64 - 128.0) / 127.0;
        let clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let angle = FRAC_PI_2 - const_asin(clamped);
        t[i as usize] = ((angle / PI) * 255.0) as u8;
        i += 1;
    }
    t
};

pub static ATAN_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    while i < 256 {
        let v = (i as f64 - 128.0) / 127.0;
        let angle = const_atan(v);
        t[i as usize] = (((angle + FRAC_PI_2) / PI) * 255.0) as u8;
        i += 1;
    }
    t
};

pub static LOG2_256: [u8; 256] = {
    let mut t = [0u8; 256];
    t[0] = 0;
    let max_log2 = const_log2(255.0);
    let mut i = 1u16;
    while i < 256 {
        let v = const_log2(i as f64) / max_log2;
        let scaled = v * 255.0;
        t[i as usize] = if scaled > 255.0 { 255 } else { scaled as u8 };
        i += 1;
    }
    t
};

pub static LOG10_256: [u8; 256] = {
    let mut t = [0u8; 256];
    t[0] = 0;
    let log2_10 = const_log2(10.0);
    let max_log10 = const_log2(255.0) / log2_10;
    let mut i = 1u16;
    while i < 256 {
        let v = (const_log2(i as f64) / log2_10) / max_log10;
        let scaled = v * 255.0;
        t[i as usize] = if scaled > 255.0 { 255 } else { scaled as u8 };
        i += 1;
    }
    t
};

pub static EXP2_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = i as f64 / 255.0 * 8.0;
        let v = const_exp2(x) / 256.0;
        let scaled = v * 255.0;
        t[i as usize] = if scaled > 255.0 { 255 } else { scaled as u8 };
        i += 1;
    }
    t
};

pub static EXP10_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let log2_10 = const_log2(10.0);
    let mut i = 0u16;
    while i < 256 {
        let x = i as f64 / 255.0 * 2.4;
        let v = const_exp2(x * log2_10) / 255.0;
        let scaled = v * 255.0;
        t[i as usize] = if scaled > 255.0 { 255 } else { scaled as u8 };
        i += 1;
    }
    t
};

pub static SQUARE_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = i as f64 / 255.0;
        t[i as usize] = (x * x * 255.0) as u8;
        i += 1;
    }
    t
};

pub static CUBE_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = i as f64 / 255.0;
        t[i as usize] = (x * x * x * 255.0) as u8;
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_bounds() {
        assert!((125..=131).contains(&SIN_256[0]));
        assert!(SIN_256[64] > 250);
        assert!(SIN_256[192] < 5);
    }

    #[test]
    fn cos_bounds() {
        assert!(COS_256[0] > 250);
        assert!((125..=131).contains(&COS_256[64]));
        assert!(COS_256[128] < 5);
    }

    #[test]
    fn exp2_monotonic() {
        for i in 0..255usize {
            assert!(EXP2_256[i] <= EXP2_256[i + 1]);
        }
    }

    #[test]
    fn square_endpoints() {
        assert_eq!(SQUARE_256[0], 0);
        assert_eq!(SQUARE_256[255], 255);
    }

    #[test]
    fn cube_endpoints() {
        assert_eq!(CUBE_256[0], 0);
        assert_eq!(CUBE_256[255], 255);
    }
}
