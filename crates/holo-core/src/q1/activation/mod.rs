//! Q1 activation and scientific function tables (21 tables, 128 KB each, 2.7 MB total).

pub(crate) mod math;

mod basic;
mod modern;
mod scientific;

// Re-export all tables.
pub use basic::*;
pub use modern::*;
pub use scientific::*;

// ── Inline accessor functions via macro ────────────────────────

macro_rules! lut_fn_q1 {
    ($name:ident, $table:ident) => {
        #[inline]
        pub fn $name(value: u16) -> u16 {
            $table[value as usize]
        }
    };
}

lut_fn_q1!(sigmoid_q1, SIGMOID_65536);
lut_fn_q1!(tanh_q1, TANH_65536);
lut_fn_q1!(exp_q1, EXP_65536);
lut_fn_q1!(log_q1, LOG_65536);
lut_fn_q1!(relu_q1, RELU_65536);
lut_fn_q1!(sqrt_q1, SQRT_65536);
lut_fn_q1!(abs_q1, ABS_65536);
lut_fn_q1!(gelu_q1, GELU_65536);
lut_fn_q1!(silu_q1, SILU_65536);
lut_fn_q1!(sin_q1, SIN_65536);
lut_fn_q1!(cos_q1, COS_65536);
lut_fn_q1!(tan_q1, TAN_65536);
lut_fn_q1!(asin_q1, ASIN_65536);
lut_fn_q1!(acos_q1, ACOS_65536);
lut_fn_q1!(atan_q1, ATAN_65536);
lut_fn_q1!(log2_q1, LOG2_65536);
lut_fn_q1!(log10_q1, LOG10_65536);
lut_fn_q1!(exp2_q1, EXP2_65536);
lut_fn_q1!(exp10_q1, EXP10_65536);
lut_fn_q1!(square_q1, SQUARE_65536);
lut_fn_q1!(cube_q1, CUBE_65536);

// ── Table ID constants and registry ────────────────────────────

pub const Q1_TABLE_COUNT: usize = 21;

pub const Q1_ID_SIGMOID: u8 = 0;
pub const Q1_ID_TANH: u8 = 1;
pub const Q1_ID_EXP: u8 = 2;
pub const Q1_ID_LOG: u8 = 3;
pub const Q1_ID_RELU: u8 = 4;
pub const Q1_ID_SQRT: u8 = 5;
pub const Q1_ID_ABS: u8 = 6;
pub const Q1_ID_GELU: u8 = 7;
pub const Q1_ID_SILU: u8 = 8;
pub const Q1_ID_SIN: u8 = 9;
pub const Q1_ID_COS: u8 = 10;
pub const Q1_ID_TAN: u8 = 11;
pub const Q1_ID_ASIN: u8 = 12;
pub const Q1_ID_ACOS: u8 = 13;
pub const Q1_ID_ATAN: u8 = 14;
pub const Q1_ID_LOG2: u8 = 15;
pub const Q1_ID_LOG10: u8 = 16;
pub const Q1_ID_EXP2: u8 = 17;
pub const Q1_ID_EXP10: u8 = 18;
pub const Q1_ID_SQUARE: u8 = 19;
pub const Q1_ID_CUBE: u8 = 20;

pub static Q1_TABLES: [&[u16; 65536]; Q1_TABLE_COUNT] = [
    &SIGMOID_65536,
    &TANH_65536,
    &EXP_65536,
    &LOG_65536,
    &RELU_65536,
    &SQRT_65536,
    &ABS_65536,
    &GELU_65536,
    &SILU_65536,
    &SIN_65536,
    &COS_65536,
    &TAN_65536,
    &ASIN_65536,
    &ACOS_65536,
    &ATAN_65536,
    &LOG2_65536,
    &LOG10_65536,
    &EXP2_65536,
    &EXP10_65536,
    &SQUARE_65536,
    &CUBE_65536,
];

/// Get a Q1 LUT table by numeric ID.
#[inline]
pub fn activation_table_q1_by_id(id: u8) -> Option<&'static [u16; 65536]> {
    Q1_TABLES.get(id as usize).copied()
}

/// Get Q1 table ID from activation name.
pub fn activation_table_q1_id(name: &str) -> Option<u8> {
    match name {
        "sigmoid" => Some(Q1_ID_SIGMOID),
        "tanh" => Some(Q1_ID_TANH),
        "exp" => Some(Q1_ID_EXP),
        "log" => Some(Q1_ID_LOG),
        "relu" => Some(Q1_ID_RELU),
        "sqrt" => Some(Q1_ID_SQRT),
        "abs" => Some(Q1_ID_ABS),
        "gelu" => Some(Q1_ID_GELU),
        "silu" => Some(Q1_ID_SILU),
        "sin" => Some(Q1_ID_SIN),
        "cos" => Some(Q1_ID_COS),
        "tan" => Some(Q1_ID_TAN),
        "asin" => Some(Q1_ID_ASIN),
        "acos" => Some(Q1_ID_ACOS),
        "atan" => Some(Q1_ID_ATAN),
        "log2" => Some(Q1_ID_LOG2),
        "log10" => Some(Q1_ID_LOG10),
        "exp2" => Some(Q1_ID_EXP2),
        "exp10" => Some(Q1_ID_EXP10),
        "square" => Some(Q1_ID_SQUARE),
        "cube" => Some(Q1_ID_CUBE),
        _ => None,
    }
}

/// Map Q1 activation name to its static table reference.
pub fn activation_table_q1(name: &str) -> Option<&'static [u16; 65536]> {
    activation_table_q1_id(name).and_then(activation_table_q1_by_id)
}

/// Get Q1 activation name from table ID.
pub fn activation_name_q1_by_id(id: u8) -> Option<&'static str> {
    const NAMES: [&str; Q1_TABLE_COUNT] = [
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
        assert!(core::ptr::eq(
            activation_table_q1("sigmoid").unwrap(),
            &SIGMOID_65536
        ));
        assert!(core::ptr::eq(
            activation_table_q1("sin").unwrap(),
            &SIN_65536
        ));
        assert!(activation_table_q1("unknown").is_none());
    }

    #[test]
    fn table_count_matches() {
        assert_eq!(Q1_TABLES.len(), Q1_TABLE_COUNT);
    }

    #[test]
    fn all_names_round_trip() {
        for id in 0..Q1_TABLE_COUNT as u8 {
            let name = activation_name_q1_by_id(id).unwrap();
            assert_eq!(activation_table_q1_id(name).unwrap(), id);
        }
    }

    #[test]
    fn all_tables_are_128kb() {
        for table in &Q1_TABLES {
            assert_eq!(core::mem::size_of_val(*table), 131072); // 65536 * 2
        }
    }

    #[test]
    fn total_memory_under_3mb() {
        let total = Q1_TABLE_COUNT * 131072; // 21 * 128KB
        assert!(total < 3 * 1024 * 1024, "total = {} bytes", total);
    }
}
