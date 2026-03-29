//! Accumulation pattern conformance tests.
//!
//! Tests the fundamental fused multiply-add at every quantum level.

use hologram_ring::{accumulate, QuantumLevel, RingWord, Q0, Q1, Q3, Q7};

fn assert_accumulate_at_level<Q: QuantumLevel>()
where
    Q::Word: core::fmt::Debug,
{
    let zero = Q::Word::ZERO;
    let one = Q::Word::ONE;

    // Basic: acc + a * b
    assert_eq!(
        accumulate(zero, Q::Word::from_u64(3), Q::Word::from_u64(5)),
        Q::Word::from_u64(15)
    );
    assert_eq!(
        accumulate(
            Q::Word::from_u64(10),
            Q::Word::from_u64(3),
            Q::Word::from_u64(5)
        ),
        Q::Word::from_u64(25)
    );

    // Identity: acc + x * 1 == acc + x
    let x = Q::Word::from_u64(42);
    let acc = Q::Word::from_u64(100);
    assert_eq!(accumulate(acc, x, one), acc.wrapping_add(x));

    // Zero: acc + x * 0 == acc
    assert_eq!(accumulate(acc, x, zero), acc);
    assert_eq!(accumulate(acc, zero, x), acc);
}

#[test]
fn accumulate_q0() {
    assert_accumulate_at_level::<Q0>();
}

#[test]
fn accumulate_q1() {
    assert_accumulate_at_level::<Q1>();
}

#[test]
fn accumulate_q3() {
    assert_accumulate_at_level::<Q3>();
}

#[test]
fn accumulate_q7() {
    assert_accumulate_at_level::<Q7>();
}

#[test]
fn dot_product_q3() {
    // Dot product: Σ a[i] * b[i]
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b: Vec<u32> = vec![5, 6, 7, 8];
    let mut acc = 0u32;
    for i in 0..4 {
        acc = accumulate(acc, a[i], b[i]);
    }
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert_eq!(acc, 70);
}

#[test]
fn matmul_2x2_q3() {
    // C = A * B where A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C[0,0] = 1*5 + 2*7 = 19
    // C[0,1] = 1*6 + 2*8 = 22
    // C[1,0] = 3*5 + 4*7 = 43
    // C[1,1] = 3*6 + 4*8 = 50
    let a: [[u32; 2]; 2] = [[1, 2], [3, 4]];
    let b: [[u32; 2]; 2] = [[5, 6], [7, 8]];
    let mut c = [[0u32; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                c[i][j] = accumulate(c[i][j], a[i][k], b[k][j]);
            }
        }
    }

    assert_eq!(c[0][0], 19);
    assert_eq!(c[0][1], 22);
    assert_eq!(c[1][0], 43);
    assert_eq!(c[1][1], 50);
}

#[test]
fn accumulate_wrapping_q0() {
    // Test wrapping behavior: 200 + 200*2 = 200 + 400 = 600, but mod 256 = 88
    assert_eq!(accumulate(200u8, 200u8, 2u8), 88u8);
}
