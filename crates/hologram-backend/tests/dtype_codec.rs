//! Float dtype codec round-trip tests (spec V.7).

use hologram_backend::cpu::dtype::*;

#[test]
fn f32_round_trip() {
    let mut buf = vec![0u8; 16];
    write_f32(&mut buf, 0, 1.5);
    write_f32(&mut buf, 1, -2.25);
    write_f32(&mut buf, 2, 1e10);
    write_f32(&mut buf, 3, -1e-10);
    assert_eq!(read_f32(&buf, 0), 1.5);
    assert_eq!(read_f32(&buf, 1), -2.25);
    assert!((read_f32(&buf, 2) - 1e10).abs() / 1e10 < 1e-6);
    assert!((read_f32(&buf, 3) + 1e-10).abs() / 1e-10 < 1e-6);
}

#[test]
fn bf16_round_trip() {
    // bf16 truncates the lower 16 bits, so exact f32 round-trip only works
    // when those bits are zero. We pick test values with that property.
    let mut buf = vec![0u8; 8];
    write_bf16(&mut buf, 0, 1.0);
    write_bf16(&mut buf, 1, -2.0);
    write_bf16(&mut buf, 2, 0.5);
    write_bf16(&mut buf, 3, 0.0);
    assert_eq!(read_bf16(&buf, 0), 1.0);
    assert_eq!(read_bf16(&buf, 1), -2.0);
    assert_eq!(read_bf16(&buf, 2), 0.5);
    assert_eq!(read_bf16(&buf, 3), 0.0);
}

#[test]
fn f16_round_trip_simple() {
    // f16 round-trip for values within representable range.
    let mut buf = vec![0u8; 8];
    write_f16(&mut buf, 0, 1.0);
    write_f16(&mut buf, 1, 0.5);
    write_f16(&mut buf, 2, -2.0);
    write_f16(&mut buf, 3, 0.0);
    assert!((read_f16(&buf, 0) - 1.0).abs() < 1e-3);
    assert!((read_f16(&buf, 1) - 0.5).abs() < 1e-3);
    assert!((read_f16(&buf, 2) + 2.0).abs() < 1e-3);
    assert_eq!(read_f16(&buf, 3), 0.0);
}

#[test]
fn read_float_dispatches_on_dtype() {
    let mut buf = vec![0u8; 4];
    write_f32(&mut buf, 0, 3.5);
    assert_eq!(read_float(&buf, 0, DTYPE_F32), 3.5);
}

#[test]
fn bytes_per_element_table() {
    assert_eq!(bytes_per_element(DTYPE_BOOL), 1);
    assert_eq!(bytes_per_element(DTYPE_U8), 1);
    assert_eq!(bytes_per_element(DTYPE_F16), 2);
    assert_eq!(bytes_per_element(DTYPE_BF16), 2);
    assert_eq!(bytes_per_element(DTYPE_F32), 4);
    assert_eq!(bytes_per_element(DTYPE_F64), 8);
}

#[test]
fn is_float_table() {
    assert!(is_float(DTYPE_F32));
    assert!(is_float(DTYPE_F16));
    assert!(is_float(DTYPE_BF16));
    assert!(is_float(DTYPE_F64));
    assert!(!is_float(DTYPE_U8));
    assert!(!is_float(DTYPE_I32));
    assert!(!is_float(DTYPE_BOOL));
}
