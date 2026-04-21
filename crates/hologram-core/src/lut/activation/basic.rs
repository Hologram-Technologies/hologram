//! Basic activation tables: sigmoid, tanh, exp, log, relu, sqrt, abs.

use super::math::signed;

pub static SIGMOID_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = signed(i);
        t[i as usize] = if x <= -64 {
            0
        } else if x >= 64 {
            255
        } else {
            ((x + 64) * 2) as u8
        };
        i += 1;
    }
    t
};

pub static TANH_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = signed(i);
        t[i as usize] = if x <= -64 {
            0
        } else if x >= 64 {
            255
        } else {
            ((x + 64) * 2) as u8
        };
        i += 1;
    }
    t
};

pub static EXP_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = signed(i);
        t[i as usize] = if x <= -64 {
            let u = (x + 128) as u16;
            ((u * 13) / 64) as u8
        } else if x >= 64 {
            255
        } else if x < 0 {
            let u = (x + 64) as u16;
            (13 + (u * 115) / 64) as u8
        } else {
            (128 + (x as u16 * 127) / 64) as u8
        };
        i += 1;
    }
    t
};

pub static LOG_256: [u8; 256] = {
    let mut t = [0u8; 256];
    t[0] = 0;
    let mut i = 1u16;
    while i < 256 {
        let x = i as u8;
        let log2_floor = 7 - x.leading_zeros() as u8;
        let frac = if log2_floor > 0 {
            ((x >> (log2_floor - 1)) & 1) as u16
        } else {
            0
        };
        let v = ((log2_floor as u16 * 2 + frac) * 255) / 16;
        t[i as usize] = if v > 255 { 255 } else { v as u8 };
        i += 1;
    }
    t
};

pub static RELU_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = if i < 128 { i as u8 } else { 0 };
        i += 1;
    }
    t
};

#[allow(clippy::manual_div_ceil)]
pub static SQRT_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let x = i;
        let mut low = 0u16;
        let mut high = 16u16;
        while low < high {
            let mid = (low + high + 1) / 2;
            if mid * mid <= x {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let remainder = x - low * low;
        let next_diff = 2 * low + 1;
        let frac = match (remainder * 16).checked_div(next_diff) {
            Some(v) => v,
            None => 0,
        };
        let scaled = low * 16 + frac;
        t[i as usize] = if scaled > 255 { 255 } else { scaled as u8 };
        i += 1;
    }
    t
};

pub static ABS_256: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        t[i as usize] = if i < 128 { i as u8 } else { (256 - i) as u8 };
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert_eq!(SIGMOID_256[0], 128);
        assert_eq!(SIGMOID_256[64], 255);
        assert_eq!(SIGMOID_256[128], 0);
    }

    #[test]
    fn tanh_bounds() {
        assert_eq!(TANH_256[0], 128);
        assert_eq!(TANH_256[64], 255);
        assert_eq!(TANH_256[128], 0);
    }

    #[test]
    fn relu_values() {
        for i in 0..128u8 {
            assert_eq!(RELU_256[i as usize], i);
        }
        for i in 128..=255u8 {
            assert_eq!(RELU_256[i as usize], 0);
        }
    }

    #[test]
    fn abs_values() {
        for i in 0..128u8 {
            assert_eq!(ABS_256[i as usize], i);
        }
        assert_eq!(ABS_256[255], 1);
        assert_eq!(ABS_256[128], 128);
    }

    #[test]
    fn log_monotonic() {
        for i in 1..255usize {
            assert!(LOG_256[i] <= LOG_256[i + 1]);
        }
    }

    #[test]
    fn sqrt_monotonic() {
        for i in 0..255usize {
            assert!(SQRT_256[i] <= SQRT_256[i + 1]);
        }
    }
}
