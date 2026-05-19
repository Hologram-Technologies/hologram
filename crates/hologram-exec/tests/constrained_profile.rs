//! Integration tests for constrained execution profile.
//!
//! Tests tape validation, weight window enforcement, constrained execution
//! output correctness, and region packing.

use hologram_exec::constrained::{
    ConstrainedProfile, KernelAllowlist, KernelDiscriminant, PackedWeightSpan, RegionIndex,
    WeightPolicy, WeightWindow,
};
use hologram_exec::error::ExecError;

// ── Tape validation integration ──────────────────────────────────────────────

#[test]
fn inference_allowlist_covers_all_ai_kernels() {
    use KernelDiscriminant::*;
    let ai_kernels = [
        InlineRmsNorm,
        InlineSoftmax,
        InlineAttention,
        InlineRoPE,
        KvWrite,
        KvRead,
        MatMulLut4,
        MatMulLut8,
        MatMulLut2,
        MatMulLut16,
        InlineMatMul,
        InlineGemm,
        InlineConv2d,
        InlineMaxPool2d,
        InlineAvgPool2d,
        InlineLayerNorm,
        InlineGroupNorm,
        InlineInstanceNorm,
        InlineEmbed,
        Output,
        Passthrough,
    ];
    let infer = KernelAllowlist::inference();
    for disc in &ai_kernels {
        assert!(
            format!("{infer:?}").contains(&format!("{disc:?}")),
            "inference preset missing {disc:?}"
        );
    }
}

#[test]
fn compute_allowlist_excludes_ai_specific_ops() {
    let allowlist = KernelAllowlist::compute();
    use KernelDiscriminant::*;
    let ai_only = [KvWrite, KvRead, InlineAttention, InlineRoPE, InlineEmbed];
    for disc in &ai_only {
        assert!(
            !format!("{allowlist:?}").contains(&format!("{disc:?}")),
            "compute preset should not include AI-specific {disc:?}"
        );
    }
}

#[test]
fn default_profile_has_bounded_window() {
    let p = ConstrainedProfile::default();
    assert_eq!(p.weight_policy, WeightPolicy::BoundedWindow);
    assert!(!p.allow_custom_ops);
    assert!(!p.allow_fallback_kernels);
    assert!(p.kernel_allowlist.is_none());
}

// ── Weight window integration ────────────────────────────────────────────────

#[test]
fn weight_window_sliding_eviction() {
    use hologram_graph::constant::{ConstantData, ConstantId, ConstantStore};

    let mut store = ConstantStore::new();
    // 4 constants, 256 bytes each.
    for _ in 0..4 {
        store.insert(ConstantData::Deferred {
            byte_size: 256,
            source_id: 0,
        });
    }

    let mut ww = WeightWindow::new(512); // fits 2 constants

    // Load 0, 1 → fills window.
    ww.ensure(&[ConstantId::new(0), ConstantId::new(1)], &store)
        .unwrap();
    assert_eq!(ww.current_usage(), 512);

    // Load 2 → evicts 0.
    ww.ensure(&[ConstantId::new(2)], &store).unwrap();
    assert_eq!(ww.current_usage(), 512);

    // Load 3 → evicts 1.
    ww.ensure(&[ConstantId::new(3)], &store).unwrap();
    assert_eq!(ww.current_usage(), 512);

    // Explicit evict 2 → usage drops.
    ww.evict(&[ConstantId::new(2)]);
    assert_eq!(ww.current_usage(), 256);

    // Explicit evict 3 → empty.
    ww.evict(&[ConstantId::new(3)]);
    assert_eq!(ww.current_usage(), 0);
}

#[test]
fn weight_window_rejects_oversized_constant() {
    use hologram_graph::constant::{ConstantData, ConstantId, ConstantStore};

    let mut store = ConstantStore::new();
    store.insert(ConstantData::Deferred {
        byte_size: 1024,
        source_id: 0,
    });

    let mut ww = WeightWindow::new(512);
    let err = ww.ensure(&[ConstantId::new(0)], &store).unwrap_err();
    assert!(
        matches!(err, ExecError::ConstrainedViolation(_)),
        "expected ConstrainedViolation, got {err:?}"
    );
}

// ── Region packing integration ───────────────────────────────────────────────

#[test]
fn region_index_full_lifecycle() {
    use hologram_graph::constant::ConstantId;

    let mut idx = RegionIndex::new();

    // Region 0: two weights, sequential offsets.
    idx.insert(
        ConstantId::new(0),
        PackedWeightSpan {
            offset: 0,
            len: 1000,
            region_id: 0,
        },
    );
    idx.insert(
        ConstantId::new(1),
        PackedWeightSpan {
            offset: 1000,
            len: 2000,
            region_id: 0,
        },
    );

    // Region 1: one weight.
    idx.insert(
        ConstantId::new(2),
        PackedWeightSpan {
            offset: 3000,
            len: 500,
            region_id: 1,
        },
    );

    assert_eq!(idx.n_regions(), 2);

    // Region 0 covers bytes [0, 3000).
    assert_eq!(idx.region_byte_range(0), Some((0, 3000)));
    // Region 1 covers bytes [3000, 3500).
    assert_eq!(idx.region_byte_range(1), Some((3000, 3500)));

    // Constants in region 0 are sorted by offset.
    let r0 = idx.constants_in_region(0);
    assert_eq!(r0.len(), 2);
    assert_eq!(r0[0].1.offset, 0);
    assert_eq!(r0[1].1.offset, 1000);

    // Lookup works.
    let span = idx.get(ConstantId::new(1)).unwrap();
    assert_eq!(span.offset, 1000);
    assert_eq!(span.len, 2000);
    assert_eq!(span.region_id, 0);
}

#[test]
fn region_prefetch_and_release_do_not_panic() {
    use hologram_graph::constant::ConstantId;

    let mut idx = RegionIndex::new();
    idx.insert(
        ConstantId::new(0),
        PackedWeightSpan {
            offset: 0,
            len: 100,
            region_id: 0,
        },
    );

    let weights = vec![0u8; 200];
    idx.prefetch_region(0, &weights);
    idx.release_region(0, &weights);
    // Non-existent region — should be no-op.
    idx.prefetch_region(99, &weights);
    idx.release_region(99, &weights);
}

// ── Weight policy variants ───────────────────────────────────────────────────

#[test]
fn weight_policy_enum_exhaustive() {
    // Ensure all variants can be compared and debug-printed.
    let policies = [
        WeightPolicy::FullResident,
        WeightPolicy::LazyCache,
        WeightPolicy::BoundedWindow,
        WeightPolicy::NoCacheStream,
    ];
    for p in &policies {
        let _ = format!("{p:?}");
    }
    assert_ne!(WeightPolicy::FullResident, WeightPolicy::BoundedWindow);
    assert_eq!(WeightPolicy::LazyCache, WeightPolicy::LazyCache);
}

// ── Profile builder patterns ─────────────────────────────────────────────────

#[test]
fn profile_with_compute_allowlist() {
    let profile = ConstrainedProfile {
        kernel_allowlist: Some(KernelAllowlist::compute()),
        max_weight_bytes: 128 * 1024 * 1024,
        max_activation_bytes: 32 * 1024 * 1024,
        ..Default::default()
    };
    assert_eq!(profile.weight_policy, WeightPolicy::BoundedWindow);
    assert!(profile.kernel_allowlist.is_some());
}

#[test]
fn profile_with_full_resident_no_allowlist() {
    let profile = ConstrainedProfile {
        weight_policy: WeightPolicy::FullResident,
        kernel_allowlist: None,
        allow_custom_ops: true,
        ..Default::default()
    };
    assert_eq!(profile.weight_policy, WeightPolicy::FullResident);
    assert!(profile.allow_custom_ops);
    assert!(profile.kernel_allowlist.is_none());
}

// ── Custom allowlist construction ────────────────────────────────────────────

#[test]
fn custom_allowlist_from_discriminants() {
    use hologram_exec::constrained::KernelDiscriminant::*;
    use std::collections::HashSet;

    // Minimal allowlist: just add + output.
    let allowlist =
        KernelAllowlist::from_discriminants(HashSet::from([InlineAdd, InlineMul, Output]));

    // Create a profile with this allowlist.
    let _profile = ConstrainedProfile {
        kernel_allowlist: Some(allowlist),
        ..Default::default()
    };
}
