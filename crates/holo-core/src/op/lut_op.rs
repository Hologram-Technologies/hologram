//! LutOp: 21+ activation/scientific function operations via precomputed tables.

use crate::lut::activation;

/// Activation and scientific function operations via LUT.
///
/// Each variant maps to a precomputed 256-entry table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum LutOp {
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Relu,
    Sqrt,
    Abs,
    Gelu,
    Silu,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Log2,
    Log10,
    Exp2,
    Exp10,
    Square,
    Cube,
}

impl LutOp {
    /// Apply this function to a byte via table lookup — O(1).
    #[inline]
    #[must_use]
    pub fn apply(&self, x: u8) -> u8 {
        self.table()[x as usize]
    }

    /// The precomputed table for this operation.
    #[must_use]
    pub fn table(&self) -> &'static [u8; 256] {
        activation::activation_table_by_id(self.table_id()).unwrap()
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Relu => "relu",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
            Self::Gelu => "gelu",
            Self::Silu => "silu",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Exp2 => "exp2",
            Self::Exp10 => "exp10",
            Self::Square => "square",
            Self::Cube => "cube",
        }
    }

    /// Table ID matching `activation::LUT_ID_*` constants.
    #[must_use]
    pub const fn table_id(&self) -> u8 {
        match self {
            Self::Sigmoid => activation::LUT_ID_SIGMOID,
            Self::Tanh => activation::LUT_ID_TANH,
            Self::Exp => activation::LUT_ID_EXP,
            Self::Log => activation::LUT_ID_LOG,
            Self::Relu => activation::LUT_ID_RELU,
            Self::Sqrt => activation::LUT_ID_SQRT,
            Self::Abs => activation::LUT_ID_ABS,
            Self::Gelu => activation::LUT_ID_GELU,
            Self::Silu => activation::LUT_ID_SILU,
            Self::Sin => activation::LUT_ID_SIN,
            Self::Cos => activation::LUT_ID_COS,
            Self::Tan => activation::LUT_ID_TAN,
            Self::Asin => activation::LUT_ID_ASIN,
            Self::Acos => activation::LUT_ID_ACOS,
            Self::Atan => activation::LUT_ID_ATAN,
            Self::Log2 => activation::LUT_ID_LOG2,
            Self::Log10 => activation::LUT_ID_LOG10,
            Self::Exp2 => activation::LUT_ID_EXP2,
            Self::Exp10 => activation::LUT_ID_EXP10,
            Self::Square => activation::LUT_ID_SQUARE,
            Self::Cube => activation::LUT_ID_CUBE,
        }
    }

    /// All LutOp variants.
    pub const ALL: [LutOp; 21] = [
        Self::Sigmoid,
        Self::Tanh,
        Self::Exp,
        Self::Log,
        Self::Relu,
        Self::Sqrt,
        Self::Abs,
        Self::Gelu,
        Self::Silu,
        Self::Sin,
        Self::Cos,
        Self::Tan,
        Self::Asin,
        Self::Acos,
        Self::Atan,
        Self::Log2,
        Self::Log10,
        Self::Exp2,
        Self::Exp10,
        Self::Square,
        Self::Cube,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ops_have_tables() {
        for op in &LutOp::ALL {
            let table = op.table();
            assert_eq!(table.len(), 256);
        }
    }

    #[test]
    fn apply_matches_table() {
        for op in &LutOp::ALL {
            let table = op.table();
            for i in 0..=255u8 {
                assert_eq!(op.apply(i), table[i as usize]);
            }
        }
    }

    #[test]
    fn all_names_unique() {
        let mut names: [&str; 21] = [""; 21];
        for (i, op) in LutOp::ALL.iter().enumerate() {
            names[i] = op.name();
        }
        for i in 0..21 {
            for j in (i + 1)..21 {
                assert_ne!(names[i], names[j]);
            }
        }
    }

    #[test]
    fn table_ids_unique() {
        let mut ids = [0u8; 21];
        for (i, op) in LutOp::ALL.iter().enumerate() {
            ids[i] = op.table_id();
        }
        for i in 0..21 {
            for j in (i + 1)..21 {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }
}
