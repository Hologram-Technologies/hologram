//! Const-compatible math helpers for compile-time table generation.

pub(crate) const fn const_reduce_to_pi(mut x: f64) -> f64 {
    const TAU: f64 = core::f64::consts::TAU;
    const PI: f64 = core::f64::consts::PI;
    x = x - ((x / TAU) as i64 as f64) * TAU;
    if x > PI {
        x -= TAU;
    } else if x < -PI {
        x += TAU;
    }
    x
}

const fn sin_taylor(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0)))
}

pub(crate) const fn const_sin(x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    let x = const_reduce_to_pi(x);
    if x > FRAC_PI_2 {
        sin_taylor(PI - x)
    } else if x < -FRAC_PI_2 {
        sin_taylor(-PI - x)
    } else {
        sin_taylor(x)
    }
}

pub(crate) const fn const_cos(x: f64) -> f64 {
    const_sin(x + core::f64::consts::FRAC_PI_2)
}

pub(crate) const fn const_atan(x: f64) -> f64 {
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    if x > 1.0 {
        FRAC_PI_2 - const_atan(1.0 / x)
    } else if x < -1.0 {
        -FRAC_PI_2 - const_atan(1.0 / x)
    } else {
        let x2 = x * x;
        x * (15.0 + 4.0 * x2) / (15.0 + 9.0 * x2)
    }
}

pub(crate) const fn const_sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = if x > 1.0 { x / 2.0 } else { x };
    let mut i = 0;
    while i < 20 {
        guess = (guess + x / guess) * 0.5;
        i += 1;
    }
    guess
}

pub(crate) const fn const_asin(x: f64) -> f64 {
    const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
    if x >= 1.0 {
        return FRAC_PI_2;
    }
    if x <= -1.0 {
        return -FRAC_PI_2;
    }
    let denom = const_sqrt_f64(1.0 - x * x);
    if denom < 1e-10 {
        return if x >= 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }
    const_atan(x / denom)
}

pub(crate) const fn const_log2(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut e = 0i32;
    let mut m = x;
    while m >= 2.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 1.0 {
        m *= 2.0;
        e -= 1;
    }
    let t = m - 1.0;
    const LN2_INV: f64 = core::f64::consts::LOG2_E;
    let log2_m = LN2_INV * (t - t * t / 2.0 + t * t * t / 3.0 - t * t * t * t / 4.0);
    e as f64 + log2_m
}

pub(crate) const fn const_exp2(x: f64) -> f64 {
    if x <= -20.0 {
        return 0.0;
    }
    if x >= 20.0 {
        return 1048576.0;
    }
    let int_part = x as i32;
    let frac = x - int_part as f64;
    let t = frac * core::f64::consts::LN_2;
    let exp_frac = 1.0 + t * (1.0 + t / 2.0 * (1.0 + t / 3.0 * (1.0 + t / 4.0 * (1.0 + t / 5.0))));
    let mut result = exp_frac;
    if int_part >= 0 {
        let mut i = 0;
        while i < int_part {
            result *= 2.0;
            i += 1;
        }
    } else {
        let mut i = 0;
        while i < -int_part {
            result *= 0.5;
            i += 1;
        }
    }
    result
}

/// Signed interpretation of a byte index: 0-127 positive, 128-255 negative.
pub(crate) const fn signed(i: u16) -> i16 {
    if i < 128 {
        i as i16
    } else {
        i as i16 - 256
    }
}
