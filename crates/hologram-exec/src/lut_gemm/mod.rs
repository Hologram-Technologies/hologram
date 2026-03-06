//! LUT-GEMM: matrix multiplication via quantized weight lookup tables.
//!
//! Replaces O(k) multiply-accumulate per output element with O(Q) lookups
//! where Q is the number of quantization levels (16 for 4-bit, 256 for 8-bit).
//!
//! ## Algorithm
//!
//! For C = A × W where W is quantized to Q levels:
//! 1. K-means clusters weights into Q centroids (compile-time)
//! 2. For each output element C[i,j]:
//!    - Build Psumbook: `sums[q] = Σ A[i,l]` for all l where `index[l,j] == q`
//!    - Compute: `C[i,j] = Σ sums[q] × centroid[q]`

pub mod matmul;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod psumbook;
pub mod quantize;

pub use matmul::{lut_gemm, lut_gemm_4bit, lut_gemm_8bit, max_relative_error, naive_matmul};
#[cfg(feature = "parallel")]
pub use parallel::{lut_gemm_4bit_par, lut_gemm_8bit_par, lut_gemm_par};
pub use psumbook::{Psumbook4, Psumbook8, Q4_LEVELS, Q8_LEVELS};
pub use quantize::{
    dequantize_error_q4, dequantize_error_q8, get_q4_index, pack_q4, quantize_4bit, quantize_8bit,
    quantize_auto, unpack_q4, QuantizedWeights, QuantizedWeights4, QuantizedWeights8,
};
