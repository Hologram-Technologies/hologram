//! Encoding and LUT FFI functions.

use crate::error::set_last_error;

/// Embed a continuous value into a byte using the given encoding.
///
/// `encoding_id`: 0=Angle, 1=Signed, 2=Unsigned, 3=Raw.
/// Returns the embedded byte, or 0 on error (check `holo_last_error`).
#[no_mangle]
pub extern "C" fn holo_encoding_embed(encoding_id: i32, value: f64) -> u8 {
    match get_encoding(encoding_id) {
        Some(enc) => enc.embed(value),
        None => 0,
    }
}

/// Lift a byte back to a continuous value using the given encoding.
///
/// `encoding_id`: 0=Angle, 1=Signed, 2=Unsigned, 3=Raw.
/// Returns the lifted value, or `NaN` on error.
#[no_mangle]
pub extern "C" fn holo_encoding_lift(encoding_id: i32, byte: u8) -> f64 {
    match get_encoding(encoding_id) {
        Some(enc) => enc.lift(byte),
        None => f64::NAN,
    }
}

/// Apply a LUT operation to a byte.
///
/// `lut_op`: discriminant index into `LutOp::ALL` (0..20).
/// Returns the result byte, or 0 on error.
#[no_mangle]
pub extern "C" fn holo_lut_apply(lut_op: i32, byte: u8) -> u8 {
    use holo_core::op::LutOp;
    match LutOp::ALL.get(lut_op as usize) {
        Some(op) => op.apply(byte),
        None => {
            set_last_error(format!("unknown LutOp: {lut_op}"));
            0
        }
    }
}

/// Apply a unary primitive operation to a byte.
///
/// `prim_op`: 0=Neg, 1=Bnot, 2=Succ, 3=Pred.
#[no_mangle]
pub extern "C" fn holo_prim_apply_unary(prim_op: i32, byte: u8) -> u8 {
    use holo_core::op::PrimOp;
    let ops = [PrimOp::Neg, PrimOp::Bnot, PrimOp::Succ, PrimOp::Pred];
    match ops.get(prim_op as usize) {
        Some(op) => op.apply_unary(byte),
        None => {
            set_last_error(format!("unknown unary PrimOp: {prim_op}"));
            0
        }
    }
}

/// Apply a binary primitive operation to two bytes.
///
/// `prim_op`: 4=Add, 5=Sub, 6=Mul, 7=Xor, 8=And, 9=Or.
#[no_mangle]
pub extern "C" fn holo_prim_apply_binary(prim_op: i32, a: u8, b: u8) -> u8 {
    use holo_core::op::PrimOp;
    let all = [
        PrimOp::Neg,
        PrimOp::Bnot,
        PrimOp::Succ,
        PrimOp::Pred,
        PrimOp::Add,
        PrimOp::Sub,
        PrimOp::Mul,
        PrimOp::Xor,
        PrimOp::And,
        PrimOp::Or,
    ];
    match all.get(prim_op as usize) {
        Some(op) if op.arity() == 2 => op.apply_binary(a, b),
        Some(_) => {
            set_last_error(format!("PrimOp {prim_op} is not binary"));
            0
        }
        None => {
            set_last_error(format!("unknown PrimOp: {prim_op}"));
            0
        }
    }
}

/// Resolve an encoding ID to a boxed Encoding trait object.
fn get_encoding(id: i32) -> Option<Box<dyn holo_core::encoding::Encoding>> {
    use holo_core::encoding::*;
    match id {
        0 => Some(Box::new(AngleEncoding)),
        1 => Some(Box::new(SignedEncoding)),
        2 => Some(Box::new(UnsignedEncoding)),
        3 => Some(Box::new(RawEncoding)),
        _ => {
            set_last_error(format!("unknown encoding: {id}"));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_round_trip() {
        let byte = holo_encoding_embed(0, std::f64::consts::PI);
        let val = holo_encoding_lift(0, byte);
        assert!((val - std::f64::consts::PI).abs() < 0.05);
    }

    #[test]
    fn signed_round_trip() {
        let byte = holo_encoding_embed(1, 0.5);
        let val = holo_encoding_lift(1, byte);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn unsigned_round_trip() {
        let byte = holo_encoding_embed(2, 0.5);
        let val = holo_encoding_lift(2, byte);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn raw_encoding() {
        assert_eq!(holo_encoding_embed(3, 42.0), 42);
        assert_eq!(holo_encoding_lift(3, 42) as u8, 42);
    }

    #[test]
    fn invalid_encoding() {
        let byte = holo_encoding_embed(99, 1.0);
        assert_eq!(byte, 0);
        let val = holo_encoding_lift(99, 0);
        assert!(val.is_nan());
    }

    #[test]
    fn lut_sigmoid() {
        // Sigmoid is deterministic: same input always gives same output
        let a = holo_lut_apply(0, 128);
        let b = holo_lut_apply(0, 128);
        assert_eq!(a, b);
    }

    #[test]
    fn lut_invalid() {
        let result = holo_lut_apply(99, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn prim_unary_neg() {
        let result = holo_prim_apply_unary(0, 1);
        assert_eq!(result, 255); // wrapping neg
    }

    #[test]
    fn prim_binary_add() {
        let result = holo_prim_apply_binary(4, 10, 20);
        assert_eq!(result, 30);
    }

    #[test]
    fn prim_binary_on_unary_errors() {
        let result = holo_prim_apply_binary(0, 1, 2);
        assert_eq!(result, 0); // Neg is unary, should error
    }

    #[test]
    fn prim_invalid_index() {
        assert_eq!(holo_prim_apply_unary(99, 0), 0);
        assert_eq!(holo_prim_apply_binary(99, 0, 0), 0);
    }
}
