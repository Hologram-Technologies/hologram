//! Bit-identity guard for the bf16/f16 **2-op matmul epilogue** fusion.
//!
//! The executor's `fuse_matmul_epilogue` pass now fuses `matmul → activation`
//! and `matmul → residual-add` for bf16/f16 (not just f32). That is only sound
//! if the fused kernel is **byte-identical** to running the two kernels
//! separately — otherwise fusing would silently change a model's output. The
//! equivalence holds because the fused kernel narrows the matmul product to the
//! storage dtype and reads it back before the epilogue (one rounding, exactly
//! as the unfused chain: the activation LUT and the add kernel both re-widen the
//! stored bf16/f16). These tests assert that equality directly at the kernel
//! layer, for every activation the epilogue can carry.
//!
//! The 3-op `matmul → add → activation` fusion is deliberately NOT extended to
//! bf16/f16 (it would skip narrowing the sum before the activation); this file
//! documents that by construction — it only exercises the 2-op forms.

use hologram_backend::cpu::dtype::{write_bf16, write_f16, DTYPE_BF16, DTYPE_F16};
use hologram_backend::{
    fused_activation, Backend, BinaryCall, BufferRef, CpuBackend, KernelCall, MatMulActivationCall,
    MatMulAddCall, MatMulCall, SplitReads, UnaryCall, Workspace,
};

struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let s = b.slot as usize;
        let n = self.slots[s].len();
        &mut self.slots[s][..n]
    }
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])> {
        let w = write.slot as usize;
        if reads.iter().any(|r| r.slot as usize == w) {
            return None;
        }
        let (lo, hi) = self.slots.split_at_mut(w);
        let (wbuf, hi_rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &hi_rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

fn buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 0,
    }
}

/// Deterministic pseudo-random f32 in [-1, 1).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

type Wr = fn(&mut [u8], usize, f32);

fn encode(vals: &[f32], w: Wr) -> Vec<u8> {
    let mut b = vec![0u8; vals.len() * 2];
    for (i, &v) in vals.iter().enumerate() {
        w(&mut b, i, v);
    }
    b
}

fn mm(a: u32, b: u32, out: u32, m: u32, k: u32, n: u32, dtype: u8) -> MatMulCall {
    MatMulCall {
        a: buf(a),
        b: buf(b),
        output: buf(out),
        m,
        k,
        n,
        dtype,
        b_packed: false,
    }
}

/// For each low-precision float dtype, assert `MatMulActivation(act)` is
/// byte-identical to `MatMul` then the standalone activation, for every
/// activation the fused epilogue carries.
#[test]
fn matmul_activation_bf16_f16_bit_identical_to_unfused() {
    let (m, k, n) = (3usize, 8usize, 5usize);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);

    let cases: &[(u8, Wr)] = &[(DTYPE_BF16, write_bf16 as Wr), (DTYPE_F16, write_f16 as Wr)];
    // (fused id, standalone unary variant tag)
    let acts: &[(u8, &str)] = &[
        (fused_activation::GELU, "gelu"),
        (fused_activation::SILU, "silu"),
        (fused_activation::SIGMOID, "sigmoid"),
        (fused_activation::TANH, "tanh"),
        (fused_activation::RELU, "relu"),
    ];

    for &(dt, w) in cases {
        for &(act_id, tag) in acts {
            // slots: 0=A 1=B 2=mm_out 3=act_out(unfused) 4=fused_out
            let mut ws = Ws {
                slots: vec![
                    encode(&a, w),
                    encode(&b, w),
                    vec![0u8; m * n * 2],
                    vec![0u8; m * n * 2],
                    vec![0u8; m * n * 2],
                ],
            };
            let mut be: CpuBackend<Ws> = CpuBackend::new();

            // Unfused: MatMul (slot2) then the standalone activation (slot3).
            be.dispatch(
                &KernelCall::MatMul(mm(0, 1, 2, m as u32, k as u32, n as u32, dt)),
                &mut ws,
            )
            .unwrap();
            let uc = UnaryCall {
                input: buf(2),
                output: buf(3),
                element_count: (m * n) as u64,
                witt_bits: 16,
                dtype: dt,
            };
            let act_call = match tag {
                "gelu" => KernelCall::Gelu(uc),
                "silu" => KernelCall::Silu(uc),
                "sigmoid" => KernelCall::Sigmoid(uc),
                "tanh" => KernelCall::Tanh(uc),
                "relu" => KernelCall::Relu(uc),
                _ => unreachable!(),
            };
            be.dispatch(&act_call, &mut ws).unwrap();

            // Fused: MatMulActivation into slot4.
            be.dispatch(
                &KernelCall::MatMulActivation(MatMulActivationCall {
                    mm: mm(0, 1, 4, m as u32, k as u32, n as u32, dt),
                    act: act_id,
                }),
                &mut ws,
            )
            .unwrap();

            assert_eq!(
                ws.slots[3], ws.slots[4],
                "dtype {dt} act {tag}: fused MatMulActivation != unfused MatMul+act"
            );
        }
    }
}

/// `MatMulAdd` is byte-identical to `MatMul` then a standalone elementwise add,
/// for bf16 and f16.
#[test]
fn matmul_add_bf16_f16_bit_identical_to_unfused() {
    let (m, k, n) = (4usize, 6usize, 7usize);
    let a = fill(m * k, 3);
    let b = fill(k * n, 4);
    let resid = fill(m * n, 5);

    let cases: &[(u8, Wr)] = &[(DTYPE_BF16, write_bf16 as Wr), (DTYPE_F16, write_f16 as Wr)];
    for &(dt, w) in cases {
        // slots: 0=A 1=B 2=mm_out 3=residual 4=add_out(unfused) 5=fused_out
        let mut ws = Ws {
            slots: vec![
                encode(&a, w),
                encode(&b, w),
                vec![0u8; m * n * 2],
                encode(&resid, w),
                vec![0u8; m * n * 2],
                vec![0u8; m * n * 2],
            ],
        };
        let mut be: CpuBackend<Ws> = CpuBackend::new();

        // Unfused: MatMul (slot2), then Add(slot2, slot3) -> slot4.
        be.dispatch(
            &KernelCall::MatMul(mm(0, 1, 2, m as u32, k as u32, n as u32, dt)),
            &mut ws,
        )
        .unwrap();
        be.dispatch(
            &KernelCall::Add(BinaryCall {
                a: buf(2),
                b: buf(3),
                output: buf(4),
                element_count: (m * n) as u64,
                witt_bits: 16,
                dtype: dt,
            }),
            &mut ws,
        )
        .unwrap();

        // Fused: MatMulAdd(residual = slot3) into slot5.
        be.dispatch(
            &KernelCall::MatMulAdd(MatMulAddCall {
                mm: mm(0, 1, 5, m as u32, k as u32, n as u32, dt),
                residual: buf(3),
            }),
            &mut ws,
        )
        .unwrap();

        assert_eq!(
            ws.slots[4], ws.slots[5],
            "dtype {dt}: fused MatMulAdd != unfused MatMul+Add"
        );
    }
}
