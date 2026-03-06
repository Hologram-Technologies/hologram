//! Activation and scientific function tables (21 tables, 256 bytes each).

pub(crate) mod math;

mod basic;
mod modern;
mod scientific;

// Re-export all tables.
pub use basic::*;
pub use modern::*;
pub use scientific::*;

// ── Inline accessor functions via macro ────────────────────────

macro_rules! lut_fn {
    ($name:ident, $table:ident) => {
        #[inline]
        pub const fn $name(value: u8) -> u8 {
            $table[value as usize]
        }
    };
}

lut_fn!(sigmoid_lut, SIGMOID_256);
lut_fn!(tanh_lut, TANH_256);
lut_fn!(exp_lut, EXP_256);
lut_fn!(log_lut, LOG_256);
lut_fn!(relu_lut, RELU_256);
lut_fn!(sqrt_lut, SQRT_256);
lut_fn!(abs_lut, ABS_256);
lut_fn!(gelu_lut, GELU_256);
lut_fn!(silu_lut, SILU_256);
lut_fn!(sin_lut, SIN_256);
lut_fn!(cos_lut, COS_256);
lut_fn!(tan_lut, TAN_256);
lut_fn!(asin_lut, ASIN_256);
lut_fn!(acos_lut, ACOS_256);
lut_fn!(atan_lut, ATAN_256);
lut_fn!(log2_lut, LOG2_256);
lut_fn!(log10_lut, LOG10_256);
lut_fn!(exp2_lut, EXP2_256);
lut_fn!(exp10_lut, EXP10_256);
lut_fn!(square_lut, SQUARE_256);
lut_fn!(cube_lut, CUBE_256);

// ── Table ID constants and registry ────────────────────────────

pub const LUT_TABLE_COUNT: usize = 21;

pub const LUT_ID_SIGMOID: u8 = 0;
pub const LUT_ID_TANH: u8 = 1;
pub const LUT_ID_EXP: u8 = 2;
pub const LUT_ID_LOG: u8 = 3;
pub const LUT_ID_RELU: u8 = 4;
pub const LUT_ID_SQRT: u8 = 5;
pub const LUT_ID_ABS: u8 = 6;
pub const LUT_ID_GELU: u8 = 7;
pub const LUT_ID_SILU: u8 = 8;
pub const LUT_ID_SIN: u8 = 9;
pub const LUT_ID_COS: u8 = 10;
pub const LUT_ID_TAN: u8 = 11;
pub const LUT_ID_ASIN: u8 = 12;
pub const LUT_ID_ACOS: u8 = 13;
pub const LUT_ID_ATAN: u8 = 14;
pub const LUT_ID_LOG2: u8 = 15;
pub const LUT_ID_LOG10: u8 = 16;
pub const LUT_ID_EXP2: u8 = 17;
pub const LUT_ID_EXP10: u8 = 18;
pub const LUT_ID_SQUARE: u8 = 19;
pub const LUT_ID_CUBE: u8 = 20;

pub static LUT_TABLES: [&[u8; 256]; LUT_TABLE_COUNT] = [
    &SIGMOID_256, &TANH_256, &EXP_256, &LOG_256, &RELU_256, &SQRT_256, &ABS_256, &GELU_256,
    &SILU_256, &SIN_256, &COS_256, &TAN_256, &ASIN_256, &ACOS_256, &ATAN_256, &LOG2_256,
    &LOG10_256, &EXP2_256, &EXP10_256, &SQUARE_256, &CUBE_256,
];

/// Get a LUT table by numeric ID.
#[inline]
pub fn activation_table_by_id(id: u8) -> Option<&'static [u8; 256]> {
    LUT_TABLES.get(id as usize).copied()
}

/// Get table ID from activation name.
pub fn activation_table_id(name: &str) -> Option<u8> {
    match name {
        "sigmoid" => Some(LUT_ID_SIGMOID),
        "tanh" => Some(LUT_ID_TANH),
        "exp" => Some(LUT_ID_EXP),
        "log" => Some(LUT_ID_LOG),
        "relu" => Some(LUT_ID_RELU),
        "sqrt" => Some(LUT_ID_SQRT),
        "abs" => Some(LUT_ID_ABS),
        "gelu" => Some(LUT_ID_GELU),
        "silu" => Some(LUT_ID_SILU),
        "sin" => Some(LUT_ID_SIN),
        "cos" => Some(LUT_ID_COS),
        "tan" => Some(LUT_ID_TAN),
        "asin" => Some(LUT_ID_ASIN),
        "acos" => Some(LUT_ID_ACOS),
        "atan" => Some(LUT_ID_ATAN),
        "log2" => Some(LUT_ID_LOG2),
        "log10" => Some(LUT_ID_LOG10),
        "exp2" => Some(LUT_ID_EXP2),
        "exp10" => Some(LUT_ID_EXP10),
        "square" => Some(LUT_ID_SQUARE),
        "cube" => Some(LUT_ID_CUBE),
        _ => None,
    }
}

/// Map activation name to its static table reference.
pub fn activation_table(name: &str) -> Option<&'static [u8; 256]> {
    activation_table_id(name).and_then(activation_table_by_id)
}

/// Get activation name from table ID.
pub fn activation_name_by_id(id: u8) -> Option<&'static str> {
    const NAMES: [&str; LUT_TABLE_COUNT] = [
        "sigmoid", "tanh", "exp", "log", "relu", "sqrt", "abs", "gelu", "silu", "sin", "cos",
        "tan", "asin", "acos", "atan", "log2", "log10", "exp2", "exp10", "square", "cube",
    ];
    NAMES.get(id as usize).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_table_lookup() {
        assert_eq!(activation_table("sigmoid").unwrap(), &SIGMOID_256);
        assert_eq!(activation_table("sin").unwrap(), &SIN_256);
        assert!(activation_table("unknown").is_none());
    }

    #[test]
    fn table_count_matches() {
        assert_eq!(LUT_TABLES.len(), LUT_TABLE_COUNT);
    }

    #[test]
    fn all_names_round_trip() {
        for id in 0..LUT_TABLE_COUNT as u8 {
            let name = activation_name_by_id(id).unwrap();
            assert_eq!(activation_table_id(name).unwrap(), id);
        }
    }
}
