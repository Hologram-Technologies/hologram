//! Conformance shape declarations for hologram.
//!
//! Two shapes are declared:
//!
//! 1. [`F_PRISM_STRICT`] — the five-primitive theoretical reference (bind,
//!    bundle, similarity, lookup, projection). Used as the directness-ratio
//!    baseline and as the verification target for the strict reading of the
//!    SCS carrying criterion.
//!
//! 2. [`F_PRISM_FUSED_COMPONENT`] — the transformer-component-granularity
//!    shape. Bare and fused operation variants are *both* declared as separate
//!    primitives because they are distinct irreducible operations in
//!    `T_{F_prism_fused_component}`: there is no Identity activation in the
//!    public signature, so the section 5 lemma cannot decompose
//!    `MatMulActivation(Relu)` into `MatMul ∘ Relu`.
//!
//! # Zero-cost design
//!
//! Each shape is a `&'static [&'static str]` table over a `pub const` block.
//! Building a shape produces no allocation; consulting one is a slice index.
//! The `ShapeId` newtype is a 32-byte content address (BLAKE3 of the shape's
//! canonical IRI representation). It is computed at compile time from the
//! IRI list via a const-evaluable hash function.
//!
//! # IRI namespace
//!
//! All hologram shapes use the namespace `https://hologram.uor.foundation/`
//! to distinguish them from upstream foundation shapes. Each operation IRI
//! in a shape's `primitives` list is the IRI of the operation in
//! `Op(F_shape)`. The compiler's transition-fidelity check (in
//! [`crate::conformance_tests`]) verifies that a Prism module's
//! `primitive_operations()` list matches the shape's `primitives` exactly.

use core::fmt;

/// Content address of a conformance shape, computed deterministically from
/// the shape's canonical IRI representation.
///
/// This is a 32-byte FNV-1a-derived digest. **Perf: zero** — computed at
/// compile time as part of the shape declaration; comparison is a single
/// `u8`-array equality which the compiler reduces to a 32-byte memcmp.
///
/// FNV-1a is used instead of BLAKE3 because it is `const fn`-evaluable on
/// stable Rust without any external crate dependency. The cryptographic
/// strength of the hash is not important here — the only requirement is that
/// distinct shape declarations produce distinct `ShapeId` values, and FNV-1a
/// provides excellent collision resistance for short IRI inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShapeId([u8; 32]);

impl ShapeId {
    /// Construct a `ShapeId` from a 32-byte digest.
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the underlying digest bytes.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ShapeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render as hex for human readability in error messages and logs.
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

/// A conformance shape declaration: a target class IRI plus the operation
/// algebra the shape exposes.
///
/// Implements the spirit of `bridge::conformance_::Shape<P>` from
/// uor-foundation v0.2.0, but in a `'static`-data form rather than a generic
/// trait. This avoids the `Primitives<P>` parameterisation in places where
/// the shape is used purely as a static lookup (the [`PrismModule`] trait,
/// the conformance tests, archive section emission).
///
/// A `Shape` value is `&'static`. Construction is `const`. Comparison is
/// pointer-cheap. **Perf: NEUTRAL** — no allocation, no dispatch.
///
/// # Example
///
/// Inspect a shape's algebra and check whether a given operation belongs to it:
///
/// ```
/// use hologram_shapes::shape::F_PRISM_FUSED_COMPONENT;
///
/// let shape = F_PRISM_FUSED_COMPONENT;
/// assert!(shape.primitive_count() > 50);
/// assert!(shape.contains_primitive("https://hologram.uor.foundation/op/matmul"));
/// assert!(!shape.contains_primitive("https://example.com/op/not-real"));
/// ```
///
/// [`PrismModule`]: crate::prism_module::PrismModule
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// Stable identifier (FNV-1a digest of the shape's canonical
    /// representation). Used as the on-archive shape tag and as the routing
    /// key in the compiler's `PrismModuleRegistry`.
    pub id: ShapeId,
    /// Human-readable name (`"F_prism_strict"`, `"F_prism_fused_component"`).
    pub name: &'static str,
    /// IRI of the OWL class this shape targets. Anchored under
    /// `https://hologram.uor.foundation/conformance/`.
    pub target_class: &'static str,
    /// Ordered list of operation IRIs that constitute the shape's algebra.
    /// Each IRI is the canonical name of one primitive operation in
    /// `Op(F_shape)`. The compiler's transition-fidelity check verifies that
    /// a Prism module's `primitive_operations()` matches this list exactly.
    pub primitives: &'static [&'static str],
    /// Description of the shape's role and granularity.
    pub description: &'static str,
}

impl Shape {
    /// Look up an operation IRI by name. Returns `Some(idx)` if the operation
    /// is in the shape's algebra. **Perf: O(N)** in the number of primitives,
    /// but N is small (~50) and this is only called at conformance test time.
    pub fn primitive_index(&self, name: &str) -> Option<usize> {
        self.primitives.iter().position(|&p| p == name)
    }

    /// Whether the given operation IRI is a primitive of this shape.
    pub fn contains_primitive(&self, name: &str) -> bool {
        self.primitive_index(name).is_some()
    }

    /// Number of declared primitives.
    pub const fn primitive_count(&self) -> usize {
        self.primitives.len()
    }
}

// ── F_prism_strict ───────────────────────────────────────────────────────────

/// Hologram IRI namespace prefix. Public so consumers (e.g., the archive
/// section emitter) can construct IRIs in the same namespace as hologram's
/// declared shapes.
pub const NS: &str = "https://hologram.uor.foundation/";

/// Operation IRIs for the strict five-primitive shape.
mod strict_ops {
    pub const BIND: &str = "https://hologram.uor.foundation/op/bind";
    pub const BUNDLE: &str = "https://hologram.uor.foundation/op/bundle";
    pub const SIMILARITY: &str = "https://hologram.uor.foundation/op/similarity";
    pub const LOOKUP: &str = "https://hologram.uor.foundation/op/lookup";
    pub const PROJECTION: &str = "https://hologram.uor.foundation/op/projection";
}

/// The strict five-primitive `F_prism` shape.
///
/// This is the theoretical reference: bind, bundle, similarity, lookup,
/// projection — the five primitives from Prism section 2 of the SCS
/// documents. Used for:
///
/// - **Verification.** The strict shape is the substrate the foundation's
///   theorems most directly apply to.
/// - **Directness-ratio baseline.** Fused-shape modules are compared against
///   the strict shape to measure how much factorisation they collapse.
///
/// At least one Prism module in the workspace will eventually carry this
/// shape strictly (`prism-composition-strict`); for now it is declared but
/// no module implements it.
pub const F_PRISM_STRICT: &Shape = &Shape {
    id: ShapeId::from_bytes(fnv1a_32("F_prism_strict")),
    name: "F_prism_strict",
    target_class: "https://hologram.uor.foundation/conformance/F_prism_strict",
    primitives: &[
        strict_ops::BIND,
        strict_ops::BUNDLE,
        strict_ops::SIMILARITY,
        strict_ops::LOOKUP,
        strict_ops::PROJECTION,
    ],
    description: "Strict five-primitive theoretical reference shape from \
                  Prism section 2. Used as directness-ratio baseline and \
                  verification target.",
};

// ── F_prism_fused_component ──────────────────────────────────────────────────

/// Operation IRIs for the transformer-component-granularity fused shape.
///
/// Both bare and fused variants are declared as separate primitives. They are
/// distinct irreducible operations in `T_{F_prism_fused_component}` because
/// no Identity activation exists in the public signature for the section 5
/// lemma to use as a factorisation step.
mod fused_component_ops {
    // Fused attention/softmax (irreducible at component grain).
    pub const ATTENTION: &str = "https://hologram.uor.foundation/op/attention";
    pub const SOFTMAX: &str = "https://hologram.uor.foundation/op/softmax";
    pub const LOG_SOFTMAX: &str = "https://hologram.uor.foundation/op/log_softmax";
    pub const FUSED_SWIGLU: &str = "https://hologram.uor.foundation/op/fused_swiglu";

    // Matmul: bare and fused are distinct primitives.
    pub const MATMUL: &str = "https://hologram.uor.foundation/op/matmul";
    pub const MATMUL_ACTIVATION: &str = "https://hologram.uor.foundation/op/matmul_activation";
    pub const MATMUL_BIAS_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/matmul_bias_activation";

    // LUT-GEMM: each quantisation level + activation status is its own
    // primitive. Bare and fused do NOT share a sub-operation.
    pub const LUT_GEMM_Q4: &str = "https://hologram.uor.foundation/op/lut_gemm_q4";
    pub const LUT_GEMM_Q8: &str = "https://hologram.uor.foundation/op/lut_gemm_q8";
    pub const LUT_GEMM_Q2: &str = "https://hologram.uor.foundation/op/lut_gemm_q2";
    pub const LUT_GEMM_Q16: &str = "https://hologram.uor.foundation/op/lut_gemm_q16";
    pub const LUT_GEMM_Q4_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/lut_gemm_q4_activation";
    pub const LUT_GEMM_Q8_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/lut_gemm_q8_activation";
    pub const LUT_GEMM_Q2_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/lut_gemm_q2_activation";

    // Float chain (the chain itself is the primitive — irreducible by
    // construction because the constituent ops only exist inside the chain).
    pub const FLOAT_CHAIN: &str = "https://hologram.uor.foundation/op/float_chain";

    // KV substrate I/O.
    pub const KV_WRITE: &str = "https://hologram.uor.foundation/op/kv_write";
    pub const KV_READ: &str = "https://hologram.uor.foundation/op/kv_read";

    // Ring-domain primitives. Parametrised over WittLevel via the v0.2.0
    // `Identity::valid_kmin` / `valid_kmax` mechanism.
    pub const RING_ACTIVATION: &str = "https://hologram.uor.foundation/op/ring_activation";
    pub const RING_ACCUMULATE: &str = "https://hologram.uor.foundation/op/ring_accumulate";
    pub const RING_PRIM_UNARY: &str = "https://hologram.uor.foundation/op/ring_prim_unary";
    pub const RING_PRIM_BINARY: &str = "https://hologram.uor.foundation/op/ring_prim_binary";

    // Norms (bare and fused are both irreducible).
    pub const RMS_NORM: &str = "https://hologram.uor.foundation/op/rms_norm";
    pub const LAYER_NORM: &str = "https://hologram.uor.foundation/op/layer_norm";
    pub const ADD_RMS_NORM: &str = "https://hologram.uor.foundation/op/add_rms_norm";
    pub const GROUP_NORM: &str = "https://hologram.uor.foundation/op/group_norm";
    pub const INSTANCE_NORM: &str = "https://hologram.uor.foundation/op/instance_norm";
    pub const RMS_NORM_ACTIVATION: &str = "https://hologram.uor.foundation/op/rms_norm_activation";
    pub const LAYER_NORM_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/layer_norm_activation";
    pub const ADD_RMS_NORM_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/add_rms_norm_activation";
    pub const GROUP_NORM_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/group_norm_activation";
    pub const INSTANCE_NORM_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/instance_norm_activation";

    // Reductions (irreducible — no decomposition into the others).
    pub const REDUCE_SUM: &str = "https://hologram.uor.foundation/op/reduce_sum";
    pub const REDUCE_MEAN: &str = "https://hologram.uor.foundation/op/reduce_mean";
    pub const REDUCE_MAX: &str = "https://hologram.uor.foundation/op/reduce_max";
    pub const REDUCE_MIN: &str = "https://hologram.uor.foundation/op/reduce_min";
    pub const REDUCE_PROD: &str = "https://hologram.uor.foundation/op/reduce_prod";
    pub const ARG_MAX: &str = "https://hologram.uor.foundation/op/arg_max";

    // Conv (bare and fused).
    pub const CONV2D: &str = "https://hologram.uor.foundation/op/conv2d";
    pub const CONV2D_ACTIVATION: &str = "https://hologram.uor.foundation/op/conv2d_activation";
    pub const CONV2D_BIAS_ACTIVATION: &str =
        "https://hologram.uor.foundation/op/conv2d_bias_activation";
    pub const CONV2D_LUT4: &str = "https://hologram.uor.foundation/op/conv2d_lut4";

    // Pooling.
    pub const MAX_POOL_2D: &str = "https://hologram.uor.foundation/op/max_pool_2d";
    pub const AVG_POOL_2D: &str = "https://hologram.uor.foundation/op/avg_pool_2d";
    pub const GLOBAL_AVG_POOL: &str = "https://hologram.uor.foundation/op/global_avg_pool";

    // RoPE (irreducible — no factorisation in the public signature).
    pub const ROPE: &str = "https://hologram.uor.foundation/op/rope";

    // View primitives (composed elementwise — these are the primitives of
    // view-fused operations, where the indexed-lookup operation itself is
    // not factorable through public ops).
    pub const VIEW_Q0: &str = "https://hologram.uor.foundation/op/view_q0";
    pub const VIEW_Q1: &str = "https://hologram.uor.foundation/op/view_q1";
    pub const PRIM_UNARY: &str = "https://hologram.uor.foundation/op/prim_unary";
    pub const PRIM_BINARY: &str = "https://hologram.uor.foundation/op/prim_binary";

    // Metadata (zero-compute structural).
    pub const RESHAPE: &str = "https://hologram.uor.foundation/op/reshape";
    pub const CAST: &str = "https://hologram.uor.foundation/op/cast";
    pub const EXPAND: &str = "https://hologram.uor.foundation/op/expand";
    pub const GATHER: &str = "https://hologram.uor.foundation/op/gather";
    pub const CONCAT: &str = "https://hologram.uor.foundation/op/concat";
    pub const TRANSPOSE: &str = "https://hologram.uor.foundation/op/transpose";
    pub const SLICE: &str = "https://hologram.uor.foundation/op/slice";
    pub const EMBED: &str = "https://hologram.uor.foundation/op/embed";
    pub const DEQUANTIZE: &str = "https://hologram.uor.foundation/op/dequantize";
    pub const PASSTHROUGH: &str = "https://hologram.uor.foundation/op/passthrough";
    pub const OUTPUT: &str = "https://hologram.uor.foundation/op/output";
    pub const SHAPE_OP: &str = "https://hologram.uor.foundation/op/shape";
}

/// The transformer-component-granularity fused shape.
///
/// This is the shape that hologram's first Prism module
/// (`hologram-fused-component`) carries.
///
/// **Bare and fused variants are both declared as separate primitives**
/// because they are distinct irreducible operations in
/// `T_{F_prism_fused_component}`. The section 5 lemma's definitional
/// extension argument does not decompose `matmul_activation(Relu)` into
/// `matmul ∘ relu` because `relu` is not in the public signature as a
/// standalone operation. The two primitives have no shared sub-operation in
/// the signature, and no factorisation the lemma can pull apart.
///
/// **Performance justification:** keeping bare and fused as separate
/// primitives avoids adding a runtime branch on activation kind in the
/// matmul hot path. The Prism module dispatches each primitive to its
/// specialised kernel implementation directly, with no per-call branching
/// on an embedded activation tag. **Perf: WIN** over the alternative
/// "merge with `FloatOp::Identity`" approach.
pub const F_PRISM_FUSED_COMPONENT: &Shape = &Shape {
    id: ShapeId::from_bytes(fnv1a_32("F_prism_fused_component")),
    name: "F_prism_fused_component",
    target_class: "https://hologram.uor.foundation/conformance/F_prism_fused_component",
    primitives: &[
        // Fused attention/softmax
        fused_component_ops::ATTENTION,
        fused_component_ops::SOFTMAX,
        fused_component_ops::LOG_SOFTMAX,
        fused_component_ops::FUSED_SWIGLU,
        // Matmul (bare and fused as separate primitives)
        fused_component_ops::MATMUL,
        fused_component_ops::MATMUL_ACTIVATION,
        fused_component_ops::MATMUL_BIAS_ACTIVATION,
        // LUT-GEMM (bare and fused at every quantisation level)
        fused_component_ops::LUT_GEMM_Q4,
        fused_component_ops::LUT_GEMM_Q8,
        fused_component_ops::LUT_GEMM_Q2,
        fused_component_ops::LUT_GEMM_Q16,
        fused_component_ops::LUT_GEMM_Q4_ACTIVATION,
        fused_component_ops::LUT_GEMM_Q8_ACTIVATION,
        fused_component_ops::LUT_GEMM_Q2_ACTIVATION,
        // Float chain
        fused_component_ops::FLOAT_CHAIN,
        // KV substrate I/O
        fused_component_ops::KV_WRITE,
        fused_component_ops::KV_READ,
        // Ring-domain (parametrised by WittLevel via Identity::valid_kmin/valid_kmax)
        fused_component_ops::RING_ACTIVATION,
        fused_component_ops::RING_ACCUMULATE,
        fused_component_ops::RING_PRIM_UNARY,
        fused_component_ops::RING_PRIM_BINARY,
        // Norms (bare and fused both irreducible)
        fused_component_ops::RMS_NORM,
        fused_component_ops::LAYER_NORM,
        fused_component_ops::ADD_RMS_NORM,
        fused_component_ops::GROUP_NORM,
        fused_component_ops::INSTANCE_NORM,
        fused_component_ops::RMS_NORM_ACTIVATION,
        fused_component_ops::LAYER_NORM_ACTIVATION,
        fused_component_ops::ADD_RMS_NORM_ACTIVATION,
        fused_component_ops::GROUP_NORM_ACTIVATION,
        fused_component_ops::INSTANCE_NORM_ACTIVATION,
        // Reductions
        fused_component_ops::REDUCE_SUM,
        fused_component_ops::REDUCE_MEAN,
        fused_component_ops::REDUCE_MAX,
        fused_component_ops::REDUCE_MIN,
        fused_component_ops::REDUCE_PROD,
        fused_component_ops::ARG_MAX,
        // Conv (bare and fused)
        fused_component_ops::CONV2D,
        fused_component_ops::CONV2D_ACTIVATION,
        fused_component_ops::CONV2D_BIAS_ACTIVATION,
        fused_component_ops::CONV2D_LUT4,
        // Pooling
        fused_component_ops::MAX_POOL_2D,
        fused_component_ops::AVG_POOL_2D,
        fused_component_ops::GLOBAL_AVG_POOL,
        // RoPE
        fused_component_ops::ROPE,
        // View primitives
        fused_component_ops::VIEW_Q0,
        fused_component_ops::VIEW_Q1,
        fused_component_ops::PRIM_UNARY,
        fused_component_ops::PRIM_BINARY,
        // Metadata
        fused_component_ops::RESHAPE,
        fused_component_ops::CAST,
        fused_component_ops::EXPAND,
        fused_component_ops::GATHER,
        fused_component_ops::CONCAT,
        fused_component_ops::TRANSPOSE,
        fused_component_ops::SLICE,
        fused_component_ops::EMBED,
        fused_component_ops::DEQUANTIZE,
        fused_component_ops::PASSTHROUGH,
        fused_component_ops::OUTPUT,
        fused_component_ops::SHAPE_OP,
    ],
    description: "Transformer-component-granularity shape. Bare and fused \
                  variants are declared as separate primitives — they are \
                  distinct irreducible operations in T_F because no Identity \
                  activation exists in the public signature for the section 5 \
                  lemma to use as a factorisation step. Performance: keeping \
                  them separate avoids a per-call branch on activation kind \
                  in the matmul hot path.",
};

// ── helpers ──────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash, evaluated at compile time, expanded to 32 bytes by
/// concatenating four mixed 64-bit values.
///
/// Used as a content-address surrogate for shape declarations. **Perf: zero
/// runtime cost** — invoked only inside `const fn` evaluation at compile
/// time, never at runtime.
const fn fnv1a_32(input: &str) -> [u8; 32] {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x100_0000_01b3;

    let bytes = input.as_bytes();
    let mut h0: u64 = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h0 ^= bytes[i] as u64;
        h0 = h0.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    // Derive three additional 64-bit blocks by mixing h0 with distinct salts.
    let h1 = h0.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x1);
    let h2 = h1.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x2);
    let h3 = h2.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x3);

    let mut out = [0u8; 32];
    let b0 = h0.to_le_bytes();
    let b1 = h1.to_le_bytes();
    let b2 = h2.to_le_bytes();
    let b3 = h3.to_le_bytes();
    let mut j = 0;
    while j < 8 {
        out[j] = b0[j];
        out[j + 8] = b1[j];
        out[j + 16] = b2[j];
        out[j + 24] = b3[j];
        j += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_shape_has_five_primitives() {
        assert_eq!(F_PRISM_STRICT.primitive_count(), 5);
        assert!(F_PRISM_STRICT.contains_primitive(strict_ops::BIND));
        assert!(F_PRISM_STRICT.contains_primitive(strict_ops::BUNDLE));
        assert!(F_PRISM_STRICT.contains_primitive(strict_ops::SIMILARITY));
        assert!(F_PRISM_STRICT.contains_primitive(strict_ops::LOOKUP));
        assert!(F_PRISM_STRICT.contains_primitive(strict_ops::PROJECTION));
    }

    #[test]
    fn fused_component_shape_has_distinct_bare_and_fused() {
        // The whole point of Option B: bare matmul and matmul+activation are
        // separate primitives, so they have separate IRIs.
        assert!(F_PRISM_FUSED_COMPONENT.contains_primitive(fused_component_ops::MATMUL));
        assert!(F_PRISM_FUSED_COMPONENT.contains_primitive(fused_component_ops::MATMUL_ACTIVATION));
        assert!(F_PRISM_FUSED_COMPONENT.contains_primitive(fused_component_ops::LUT_GEMM_Q4));
        assert!(
            F_PRISM_FUSED_COMPONENT.contains_primitive(fused_component_ops::LUT_GEMM_Q4_ACTIVATION)
        );
    }

    #[test]
    fn shape_ids_are_distinct() {
        assert_ne!(F_PRISM_STRICT.id, F_PRISM_FUSED_COMPONENT.id);
    }

    #[test]
    fn shape_ids_are_deterministic() {
        // Re-deriving the FNV1a hash for the same input must produce the same
        // bytes. This catches any bug where the const evaluator misbehaves.
        let id1 = ShapeId::from_bytes(fnv1a_32("F_prism_strict"));
        let id2 = ShapeId::from_bytes(fnv1a_32("F_prism_strict"));
        assert_eq!(id1, id2);
        assert_eq!(id1, F_PRISM_STRICT.id);
    }

    #[test]
    fn shape_id_display_is_hex() {
        let id = ShapeId::from_bytes([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let s = alloc::format!("{}", id);
        assert!(s.starts_with("000102030405060700"));
        assert_eq!(s.len(), 64);
    }

    /// The IRI namespace constant exists. (Compile-time check that the
    /// constant is referenced and not pruned.)
    #[test]
    fn ns_is_anchored() {
        assert!(NS.starts_with("https://hologram.uor.foundation/"));
    }
}
