//! Densified `Dequantize → activation` conformance (PM_7 generalization).
//!
//! A quantized (i8) tensor feeding a transcendental activation has a realized
//! domain of ≤256 values regardless of its f32 storage, so the composition is
//! densified into one table indexed by the quantized byte. This must be
//! **bit-identical** to the unfused `Dequantize (→ f32) → activation` pair (the
//! table entry runs the exact same f32 dequant arithmetic and reference
//! activation) — a pure speedup, not an approximation. Validated for every i8
//! value and against the f64 reference of `activation((q − zp)·scale)`.

use hologram_backend::cpu::dtype::{read_f32, DTYPE_F32, DTYPE_I8, DTYPE_U8};
use hologram_backend::{
    lut_act, Backend, BufferRef, CpuBackend, DequantActivationCall, DequantizeCall, KernelCall,
    SplitReads, UnaryCall, Workspace,
};

struct Ws {
    slots: Vec<Vec<u8>>,
}
impl Workspace for Ws {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize]
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
        let (wbuf, rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

fn rb(s: u32) -> BufferRef {
    BufferRef {
        slot: s,
        offset: 0,
        length: 0,
    }
}

fn gelu_ref(x: f64) -> f64 {
    // Same tanh approximation as `gelu_f`, evaluated in f64.
    0.5 * x * (1.0 + (0.797_884_6_f64 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Every i8 value, dequantized and activated through the fused table, must be
/// bit-identical to dequantizing to f32 then activating.
#[test]
fn dequant_gelu_table_is_bit_identical_to_unfused() {
    let n = 256usize;
    let qbytes: Vec<u8> = (0..256).map(|i| i as u8).collect(); // covers all i8 values
    let scale_bits = 0.05f32.to_bits();
    let zero_point = -7i32;

    // Fused: DequantActivation → out slot 1.
    let mut wf = Ws {
        slots: vec![qbytes.clone(), vec![0u8; n * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::DequantActivation(DequantActivationCall {
            input: rb(0),
            output: rb(1),
            element_count: n as u64,
            quant_dtype: DTYPE_I8,
            act: lut_act::GELU,
            dtype: DTYPE_F32,
            scale_bits,
            zero_point,
        }),
        &mut wf,
    )
    .unwrap();

    // Unfused: Dequantize (→ f32, slot 1) then Gelu (slot 1 → slot 2).
    let mut wu = Ws {
        slots: vec![qbytes.clone(), vec![0u8; n * 4], vec![0u8; n * 4]],
    };
    be.dispatch(
        &KernelCall::Dequantize(DequantizeCall {
            input: rb(0),
            scales: DequantizeCall::NO_VEC,
            zero_points: DequantizeCall::NO_VEC,
            output: rb(1),
            element_count: n as u64,
            channels: 0,
            inner: 0,
            quant_dtype: DTYPE_I8,
            dtype: DTYPE_F32,
            scale_bits,
            zero_point,
        }),
        &mut wu,
    )
    .unwrap();
    be.dispatch(
        &KernelCall::Gelu(UnaryCall {
            input: rb(1),
            output: rb(2),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
        &mut wu,
    )
    .unwrap();

    let scale = f32::from_bits(scale_bits);
    for i in 0..n {
        let fused = read_f32(&wf.slots[1], i);
        let unfused = read_f32(&wu.slots[2], i);
        // Bit-identical to the unfused composition.
        assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "fused != unfused at q={i}: {fused} vs {unfused}"
        );
        // And matches the f64 reference of the dequantized input.
        let q = (i as u8 as i8) as i32;
        let x = (q - zero_point) as f64 * scale as f64;
        let want = gelu_ref(x);
        assert!(
            (fused as f64 - want).abs() <= 1e-5 + 1e-5 * want.abs(),
            "gelu table q={i} x={x} got {fused} want {want}"
        );
    }
}

/// Same, for **uint8** (ONNX's default asymmetric quantization type): the
/// quantized byte is read unsigned (0..=255) with an asymmetric zero-point.
#[test]
fn dequant_gelu_table_uint8_is_bit_identical_to_unfused() {
    let n = 256usize;
    let qbytes: Vec<u8> = (0..256).map(|i| i as u8).collect(); // all u8 values
    let scale_bits = 0.03f32.to_bits();
    let zero_point = 128i32; // asymmetric, mid-range

    let mut wf = Ws {
        slots: vec![qbytes.clone(), vec![0u8; n * 4]],
    };
    let mut be: CpuBackend<Ws> = CpuBackend::new();
    be.dispatch(
        &KernelCall::DequantActivation(DequantActivationCall {
            input: rb(0),
            output: rb(1),
            element_count: n as u64,
            quant_dtype: DTYPE_U8,
            act: lut_act::GELU,
            dtype: DTYPE_F32,
            scale_bits,
            zero_point,
        }),
        &mut wf,
    )
    .unwrap();

    let mut wu = Ws {
        slots: vec![qbytes, vec![0u8; n * 4], vec![0u8; n * 4]],
    };
    be.dispatch(
        &KernelCall::Dequantize(DequantizeCall {
            input: rb(0),
            scales: DequantizeCall::NO_VEC,
            zero_points: DequantizeCall::NO_VEC,
            output: rb(1),
            element_count: n as u64,
            channels: 0,
            inner: 0,
            quant_dtype: DTYPE_U8,
            dtype: DTYPE_F32,
            scale_bits,
            zero_point,
        }),
        &mut wu,
    )
    .unwrap();
    be.dispatch(
        &KernelCall::Gelu(UnaryCall {
            input: rb(1),
            output: rb(2),
            element_count: n as u64,
            witt_bits: 32,
            dtype: DTYPE_F32,
        }),
        &mut wu,
    )
    .unwrap();

    let scale = f32::from_bits(scale_bits);
    for i in 0..n {
        let fused = read_f32(&wf.slots[1], i);
        let unfused = read_f32(&wu.slots[2], i);
        assert_eq!(
            fused.to_bits(),
            unfused.to_bits(),
            "u8 fused != unfused at q={i}: {fused} vs {unfused}"
        );
        // Unsigned byte: q = i directly (0..=255).
        let x = (i as i32 - zero_point) as f64 * scale as f64;
        let want = gelu_ref(x);
        assert!(
            (fused as f64 - want).abs() <= 1e-5 + 1e-5 * want.abs(),
            "u8 gelu table q={i} x={x} got {fused} want {want}"
        );
    }
}
