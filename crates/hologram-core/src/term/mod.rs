//! Term AST types for the UOR term language.
//!
//! Provides an arena-allocated, `no_std`-compatible term representation
//! matching the core subset of the UOR EBNF grammar:
//! `literal | application | variable`.
//!
//! All types are `Copy` and heap-free. The arena is append-only with O(1)
//! allocation and O(1) indexed access.

mod arena;
pub mod compile_unit;

pub use arena::TermArena;
pub use compile_unit::{HoloAddress, HoloCompileUnit, PreflightStatus};

use crate::op::{LutOp, PrimOp, RingLevel};

/// Index into the TermArena's float op table. Keeps TermKind small.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct FloatOpRef(pub u32);

/// Index into the TermArena's view table. Keeps TermKind small.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct ViewRef(pub u32);

/// Index into the TermArena's constant table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct ConstRef(pub u32);

/// Index into a [`TermArena`]. 32-bit for cache density (4B vs 8B pointer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct TermId(pub u32);

/// De Bruijn index for variable references. Supports up to 65535 bindings.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct VarId(pub u16);

/// Index into a type declaration table. `u16::MAX` = unconstrained.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct TypeId(pub u16);

impl TypeId {
    /// Sentinel: no type constraint.
    pub const UNCONSTRAINED: Self = Self(u16::MAX);
}

/// Discriminated union for term nodes.
///
/// Each variant corresponds to a production in the UOR EBNF grammar.
/// Largest variant (`BinaryApp`) is 10 bytes of payload; with discriminant
/// and padding the enum fits in 16 bytes (4 per 64-byte cache line).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    /// Integer literal — covers all quantum levels via `i64`.
    /// Grammar: `integer-literal ::= digit { digit }`.
    IntLit(i64),

    /// Braille address literal — single byte value in `[0, 255]`.
    /// Grammar: `braille-literal ::= braille-glyph { braille-glyph }`.
    BrailleLit(u8),

    /// Quantum-tagged literal — value at an explicit quantum level.
    /// Grammar: `quantum-literal ::= integer-literal "@" quantum-level`.
    QuantumLit { level: RingLevel, value: u32 },

    /// Unary application — `op(arg)`.
    /// Grammar: `unary-application ::= unary-op "(" term ")"`.
    UnaryApp { op: PrimOp, arg: TermId },

    /// Binary application — `op(lhs, rhs)`.
    /// Grammar: `binary-application ::= binary-op "(" term "," term ")"`.
    BinaryApp { op: PrimOp, lhs: TermId, rhs: TermId },

    /// Variable reference by de Bruijn index.
    /// Grammar: `variable ::= identifier`.
    Var(VarId),

    /// LUT activation function application — `lut-op(arg)`.
    /// Grammar: `lut-application ::= lut-op "(" term ")"`.
    LutApp { op: LutOp, arg: TermId },

    // ── Compiler IR variants (produced by lowering/fusion, not user-parseable) ──

    /// Float operation application. The `FloatOpRef` indexes into the arena's
    /// float op table (avoids bloating TermKind with large FloatOp variants).
    /// `arg1` is `TermId(u32::MAX)` for unary float ops.
    FloatApp { op: FloatOpRef, arg0: TermId, arg1: TermId },

    /// Ring-level tagged unary operation (produced by precision promotion pass).
    RingUnaryApp { op: PrimOp, level: RingLevel, arg: TermId },

    /// Ring-level tagged binary operation (produced by precision promotion pass).
    RingBinaryApp { op: PrimOp, level: RingLevel, lhs: TermId, rhs: TermId },

    /// Reference to a constant in the constant store.
    Constant(ConstRef),

    /// Graph input boundary (index into input slot list).
    GraphInput(u32),

    /// Graph output boundary — wraps the term to output.
    GraphOutput(TermId),

    /// Fused element-wise view (produced by view fusion).
    /// Indexes into the arena's view table.
    FusedViewRef(ViewRef),

    /// Identity / zero-copy forward (produced by involution cancellation).
    Passthrough(TermId),
}

/// A node in the [`TermArena`]: the term kind plus an optional type annotation.
#[derive(Clone, Copy, Debug)]
pub struct TermNode {
    pub kind: TermKind,
    /// Type constraint on this node. [`TypeId::UNCONSTRAINED`] if none.
    pub ty: TypeId,
}

/// A let-binding: `let x : T = rhs ;`
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub var: VarId,
    pub ty: TypeId,
    pub rhs: TermId,
}

/// An assertion: `assert lhs = rhs ;` or `assert lhs ≡ rhs ;`
#[derive(Clone, Copy, Debug)]
pub struct Assertion {
    pub lhs: TermId,
    pub rhs: TermId,
    /// `false` = strict ring equality (`=`), `true` = canonical-form equivalence (`≡`).
    pub canonical: bool,
}

/// Constraint kind for type declarations.
///
/// Grammar: `constraint-kind ::= "residue" | "carry" | "hamming" | "depth" | "fiber" | "affine"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ConstraintKind {
    /// Vertical / ring-arithmetic axis.
    Residue = 0,
    /// Carry-pattern constraint.
    Carry = 1,
    /// Horizontal / Hamming-metric axis.
    Hamming = 2,
    /// Diagonal / fiber-depth axis.
    Depth = 3,
    /// Explicit fiber assignment.
    Fiber = 4,
    /// Affine subspace constraint.
    Affine = 5,
}

/// A type declaration with a single constraint.
///
/// Grammar: `type-decl ::= "type" identifier "{" { constraint-decl } "}"`.
#[derive(Clone, Copy, Debug)]
pub struct TypeDecl {
    pub name_id: VarId,
    pub constraint: ConstraintKind,
    /// The constraint expression (a term reference).
    pub value: TermId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn type_sizes() {
        assert_eq!(size_of::<TermId>(), 4);
        assert_eq!(size_of::<VarId>(), 2);
        assert_eq!(size_of::<TypeId>(), 2);
        assert!(
            size_of::<TermKind>() <= 16,
            "TermKind is {} bytes, expected <= 16",
            size_of::<TermKind>()
        );
        assert!(
            size_of::<TermNode>() <= 24,
            "TermNode is {} bytes, expected <= 24",
            size_of::<TermNode>()
        );
    }

    #[test]
    fn type_id_unconstrained() {
        assert_eq!(TypeId::UNCONSTRAINED.0, u16::MAX);
    }

    #[test]
    fn term_kind_variants() {
        let lit = TermKind::IntLit(42);
        let braille = TermKind::BrailleLit(0xFF);
        let qlit = TermKind::QuantumLit {
            level: RingLevel::Q0,
            value: 42,
        };
        let unary = TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: TermId(0),
        };
        let binary = TermKind::BinaryApp {
            op: PrimOp::Add,
            lhs: TermId(0),
            rhs: TermId(1),
        };
        let var = TermKind::Var(VarId(0));

        // Ensure all variants are distinct.
        assert_ne!(lit, braille);
        assert_ne!(qlit, unary);
        assert_ne!(binary, var);
    }
}
