#[cfg(test)]
mod size_audit {
    use crate::carry::CurvatureFlux;
    use crate::op::{FloatOp, PrimOp, RingLevel};
    use crate::term::{ConstRef, FloatOpRef, TermId, TermKind, TermNode, TypeId, VarId, ViewRef};
    use std::mem::size_of;

    #[test]
    fn audit_basic_type_sizes() {
        // Reference types
        println!("\n=== Index Types ===");
        println!("TermId: {} bytes", size_of::<TermId>());
        println!("VarId: {} bytes", size_of::<VarId>());
        println!("TypeId: {} bytes", size_of::<TypeId>());
        println!("FloatOpRef: {} bytes", size_of::<FloatOpRef>());
        println!("ViewRef: {} bytes", size_of::<ViewRef>());
        println!("ConstRef: {} bytes", size_of::<ConstRef>());

        // Compound types
        println!("\n=== Compound Types ===");
        println!("TermKind: {} bytes", size_of::<TermKind>());
        println!("TermNode: {} bytes", size_of::<TermNode>());
        println!("CurvatureFlux: {} bytes", size_of::<CurvatureFlux>());
        
        println!("\n=== Op Types ===");
        println!("PrimOp: {} bytes", size_of::<PrimOp>());
        println!("RingLevel: {} bytes", size_of::<RingLevel>());
        println!("FloatOp: {} bytes", size_of::<FloatOp>());

        // Verify constraints
        assert_eq!(size_of::<TermId>(), 4, "TermId should be 4 bytes");
        assert_eq!(size_of::<VarId>(), 2, "VarId should be 2 bytes");
        assert_eq!(size_of::<TypeId>(), 2, "TypeId should be 2 bytes");
        assert!(size_of::<TermKind>() <= 16, "TermKind overflow check");
        assert!(size_of::<TermNode>() <= 24, "TermNode overflow check");
        assert_eq!(size_of::<CurvatureFlux>(), 9, "CurvatureFlux should be 9 bytes");
    }
}
