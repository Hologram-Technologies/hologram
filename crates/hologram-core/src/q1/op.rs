//! Q1 operation types: PrimOp16 (10 primitive ops) and LutOp16 (21 activation ops).

use crate::q1::activation;
use crate::q1::arith;

/// The 10 primitive operations on Z/65536Z.
///
/// Same variants as Q0 PrimOp but operates on u16 values directly (no LUT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimOp16 {
    /// neg(x) = (-x) mod 65536
    Neg,
    /// bnot(x) = 65535 ^ x
    Bnot,
    /// succ(x) = (x + 1) mod 65536
    Succ,
    /// pred(x) = (x - 1) mod 65536
    Pred,
    /// add(x, y) = (x + y) mod 65536
    Add,
    /// sub(x, y) = (x - y) mod 65536
    Sub,
    /// mul(x, y) = (x * y) mod 65536
    Mul,
    /// xor(x, y) = x ^ y
    Xor,
    /// and(x, y) = x & y
    And,
    /// or(x, y) = x | y
    Or,
}

impl PrimOp16 {
    /// Arity: 1 for unary, 2 for binary.
    #[inline]
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Neg | Self::Bnot | Self::Succ | Self::Pred => 1,
            Self::Add | Self::Sub | Self::Mul => 2,
            Self::Xor | Self::And | Self::Or => 2,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Bnot => "bnot",
            Self::Succ => "succ",
            Self::Pred => "pred",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Xor => "xor",
            Self::And => "and",
            Self::Or => "or",
        }
    }

    /// Apply a unary primitive operation.
    #[inline]
    #[must_use]
    pub const fn apply_unary(&self, x: u16) -> u16 {
        match self {
            Self::Neg => x.wrapping_neg(),
            Self::Bnot => !x,
            Self::Succ => x.wrapping_add(1),
            Self::Pred => x.wrapping_sub(1),
            _ => 0,
        }
    }

    /// Apply a binary primitive operation (wrapping arithmetic, no LUT).
    #[inline]
    #[must_use]
    pub const fn apply_binary(&self, x: u16, y: u16) -> u16 {
        match self {
            Self::Add => arith::add_q1(x, y),
            Self::Sub => arith::sub_q1(x, y),
            Self::Mul => arith::mul_q1(x, y),
            Self::Xor => x ^ y,
            Self::And => x & y,
            Self::Or => x | y,
            _ => 0,
        }
    }
}

/// Activation and scientific function operations via Q1 LUT.
///
/// Each variant maps to a precomputed 65536-entry u16 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LutOp16 {
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

impl LutOp16 {
    /// Apply this function to a u16 via table lookup — O(1).
    #[inline]
    #[must_use]
    pub fn apply(&self, x: u16) -> u16 {
        self.table()[x as usize]
    }

    /// The precomputed Q1 table for this operation.
    #[must_use]
    pub fn table(&self) -> &'static [u16; 65536] {
        activation::activation_table_q1_by_id(self.table_id()).unwrap()
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

    /// Table ID matching `Q1_ID_*` constants.
    #[must_use]
    pub const fn table_id(&self) -> u8 {
        match self {
            Self::Sigmoid => activation::Q1_ID_SIGMOID,
            Self::Tanh => activation::Q1_ID_TANH,
            Self::Exp => activation::Q1_ID_EXP,
            Self::Log => activation::Q1_ID_LOG,
            Self::Relu => activation::Q1_ID_RELU,
            Self::Sqrt => activation::Q1_ID_SQRT,
            Self::Abs => activation::Q1_ID_ABS,
            Self::Gelu => activation::Q1_ID_GELU,
            Self::Silu => activation::Q1_ID_SILU,
            Self::Sin => activation::Q1_ID_SIN,
            Self::Cos => activation::Q1_ID_COS,
            Self::Tan => activation::Q1_ID_TAN,
            Self::Asin => activation::Q1_ID_ASIN,
            Self::Acos => activation::Q1_ID_ACOS,
            Self::Atan => activation::Q1_ID_ATAN,
            Self::Log2 => activation::Q1_ID_LOG2,
            Self::Log10 => activation::Q1_ID_LOG10,
            Self::Exp2 => activation::Q1_ID_EXP2,
            Self::Exp10 => activation::Q1_ID_EXP10,
            Self::Square => activation::Q1_ID_SQUARE,
            Self::Cube => activation::Q1_ID_CUBE,
        }
    }

    /// All LutOp16 variants.
    pub const ALL: [LutOp16; 21] = [
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

/// Unified Q1 operation enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op16 {
    /// Primitive operation (direct wrapping arithmetic).
    Prim(PrimOp16),
    /// Activation/scientific function (65536-entry LUT).
    Lut(LutOp16),
}

impl Op16 {
    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Prim(p) => p.name(),
            Self::Lut(l) => l.name(),
        }
    }

    /// Arity: 1 for unary (all LutOps + unary PrimOps), 2 for binary PrimOps.
    #[must_use]
    pub const fn arity(&self) -> u8 {
        match self {
            Self::Prim(p) => p.arity(),
            Self::Lut(_) => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PrimOp16 tests ---

    #[test]
    fn prim_unary_neg() {
        assert_eq!(PrimOp16::Neg.apply_unary(0), 0);
        assert_eq!(PrimOp16::Neg.apply_unary(1), 65535);
        assert_eq!(PrimOp16::Neg.apply_unary(32768), 32768);
    }

    #[test]
    fn prim_unary_bnot() {
        assert_eq!(PrimOp16::Bnot.apply_unary(0), 65535);
        assert_eq!(PrimOp16::Bnot.apply_unary(0xAAAA), 0x5555);
    }

    #[test]
    fn prim_succ_pred_inverse() {
        for i in (0u32..=65535).step_by(1000) {
            let v = i as u16;
            let s = PrimOp16::Succ.apply_unary(v);
            let p = PrimOp16::Pred.apply_unary(s);
            assert_eq!(p, v);
        }
    }

    #[test]
    fn prim_binary_add() {
        assert_eq!(PrimOp16::Add.apply_binary(100, 200), 300);
        assert_eq!(PrimOp16::Add.apply_binary(65535, 1), 0);
    }

    #[test]
    fn prim_binary_xor() {
        assert_eq!(PrimOp16::Xor.apply_binary(0xFF00, 0x0FF0), 0xF0F0);
    }

    #[test]
    fn prim_arity() {
        assert_eq!(PrimOp16::Neg.arity(), 1);
        assert_eq!(PrimOp16::Add.arity(), 2);
        assert_eq!(PrimOp16::Xor.arity(), 2);
    }

    // --- LutOp16 tests ---

    #[test]
    fn lut_all_ops_have_tables() {
        for op in &LutOp16::ALL {
            let table = op.table();
            assert_eq!(table.len(), 65536);
        }
    }

    #[test]
    fn lut_apply_matches_table() {
        for op in &LutOp16::ALL {
            let table = op.table();
            // Sample a few values
            for i in (0u32..=65535).step_by(3000) {
                assert_eq!(op.apply(i as u16), table[i as usize]);
            }
        }
    }

    #[test]
    fn lut_all_names_unique() {
        for i in 0..21 {
            for j in (i + 1)..21 {
                assert_ne!(LutOp16::ALL[i].name(), LutOp16::ALL[j].name());
            }
        }
    }

    #[test]
    fn lut_table_ids_unique() {
        for i in 0..21 {
            for j in (i + 1)..21 {
                assert_ne!(LutOp16::ALL[i].table_id(), LutOp16::ALL[j].table_id());
            }
        }
    }

    // --- Op16 tests ---

    #[test]
    fn op16_name() {
        assert_eq!(Op16::Prim(PrimOp16::Add).name(), "add");
        assert_eq!(Op16::Lut(LutOp16::Sigmoid).name(), "sigmoid");
    }

    #[test]
    fn op16_arity() {
        assert_eq!(Op16::Prim(PrimOp16::Neg).arity(), 1);
        assert_eq!(Op16::Prim(PrimOp16::Add).arity(), 2);
        assert_eq!(Op16::Lut(LutOp16::Sigmoid).arity(), 1);
    }
}
