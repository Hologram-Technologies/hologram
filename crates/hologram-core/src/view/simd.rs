//! SIMD-accelerated table lookups for `ElementWiseView`.
//!
//! Uses `vpshufb` (AVX2) or `pshufb` (SSE4.2) to process 32 or 16 bytes
//! per iteration. The 256-entry table is split into 16 subtables of 16 bytes.
//! For each chunk, we extract high/low nibbles, iterate over subtables,
//! mask-select matching bytes, shuffle via low nibble, and OR into result.

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "x86_64", target_feature = "sse4.2"),
))]
use super::ElementWiseView;

/// AVX2 in-place apply: process 32 bytes at a time via `vpshufb`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn apply_avx2(view: &ElementWiseView, data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;

        let table = view.table();
        let len = data.len();
        let chunks = len / 32;
        let remainder = chunks * 32;

        unsafe {
            let low_mask = _mm256_set1_epi8(0x0F);

            for chunk in 0..chunks {
                let ptr = data.as_mut_ptr().add(chunk * 32);
                let input = _mm256_loadu_si256(ptr as *const __m256i);
                let lo = _mm256_and_si256(input, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(input, 4), low_mask);
                let mut result = _mm256_setzero_si256();

                for sub in 0..16u8 {
                    let base = (sub as usize) * 16;
                    let subtable = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        table.as_ptr().add(base) as *const __m128i,
                    ));
                    let match_val = _mm256_set1_epi8(sub as i8);
                    let mask = _mm256_cmpeq_epi8(hi, match_val);
                    let shuffled = _mm256_shuffle_epi8(subtable, lo);
                    result = _mm256_or_si256(result, _mm256_and_si256(mask, shuffled));
                }

                _mm256_storeu_si256(ptr as *mut __m256i, result);
            }
        }

        // Scalar remainder
        for byte in &mut data[remainder..] {
            *byte = view.apply(*byte);
        }
    }
}

/// AVX2 separate input/output apply.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn apply_to_avx2(view: &ElementWiseView, input: &[u8], output: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;

        let table = view.table();
        let len = input.len();
        let chunks = len / 32;
        let remainder = chunks * 32;

        unsafe {
            let low_mask = _mm256_set1_epi8(0x0F);

            for chunk in 0..chunks {
                let in_ptr = input.as_ptr().add(chunk * 32) as *const __m256i;
                let out_ptr = output.as_mut_ptr().add(chunk * 32) as *mut __m256i;
                let inv = _mm256_loadu_si256(in_ptr);
                let lo = _mm256_and_si256(inv, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16(inv, 4), low_mask);
                let mut result = _mm256_setzero_si256();

                for sub in 0..16u8 {
                    let base = (sub as usize) * 16;
                    let subtable = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        table.as_ptr().add(base) as *const __m128i,
                    ));
                    let match_val = _mm256_set1_epi8(sub as i8);
                    let mask = _mm256_cmpeq_epi8(hi, match_val);
                    let shuffled = _mm256_shuffle_epi8(subtable, lo);
                    result = _mm256_or_si256(result, _mm256_and_si256(mask, shuffled));
                }

                _mm256_storeu_si256(out_ptr, result);
            }
        }

        // Scalar remainder
        for i in remainder..len {
            output[i] = view.apply(input[i]);
        }
    }
}

/// SSE4.2 in-place apply: process 16 bytes at a time via `pshufb`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
pub fn apply_sse42(view: &ElementWiseView, data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;

        let table = view.table();
        let len = data.len();
        let chunks = len / 16;
        let remainder = chunks * 16;

        unsafe {
            let low_mask = _mm_set1_epi8(0x0F);

            for chunk in 0..chunks {
                let ptr = data.as_mut_ptr().add(chunk * 16);
                let input = _mm_loadu_si128(ptr as *const __m128i);
                let lo = _mm_and_si128(input, low_mask);
                let hi = _mm_and_si128(_mm_srli_epi16(input, 4), low_mask);
                let mut result = _mm_setzero_si128();

                for sub in 0..16u8 {
                    let base = (sub as usize) * 16;
                    let subtable = _mm_loadu_si128(table.as_ptr().add(base) as *const __m128i);
                    let match_val = _mm_set1_epi8(sub as i8);
                    let mask = _mm_cmpeq_epi8(hi, match_val);
                    let shuffled = _mm_shuffle_epi8(subtable, lo);
                    result = _mm_or_si128(result, _mm_and_si128(mask, shuffled));
                }

                _mm_storeu_si128(ptr as *mut __m128i, result);
            }
        }

        // Scalar remainder
        for byte in &mut data[remainder..] {
            *byte = view.apply(*byte);
        }
    }
}

/// SSE4.2 separate input/output apply.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
pub fn apply_to_sse42(view: &ElementWiseView, input: &[u8], output: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;

        let table = view.table();
        let len = input.len();
        let chunks = len / 16;
        let remainder = chunks * 16;

        unsafe {
            let low_mask = _mm_set1_epi8(0x0F);

            for chunk in 0..chunks {
                let in_ptr = input.as_ptr().add(chunk * 16) as *const __m128i;
                let out_ptr = output.as_mut_ptr().add(chunk * 16) as *mut __m128i;
                let inv = _mm_loadu_si128(in_ptr);
                let lo = _mm_and_si128(inv, low_mask);
                let hi = _mm_and_si128(_mm_srli_epi16(inv, 4), low_mask);
                let mut result = _mm_setzero_si128();

                for sub in 0..16u8 {
                    let base = (sub as usize) * 16;
                    let subtable = _mm_loadu_si128(table.as_ptr().add(base) as *const __m128i);
                    let match_val = _mm_set1_epi8(sub as i8);
                    let mask = _mm_cmpeq_epi8(hi, match_val);
                    let shuffled = _mm_shuffle_epi8(subtable, lo);
                    result = _mm_or_si128(result, _mm_and_si128(mask, shuffled));
                }

                _mm_storeu_si128(out_ptr, result);
            }
        }

        // Scalar remainder
        for i in remainder..len {
            output[i] = view.apply(input[i]);
        }
    }
}

/// ARM NEON in-place apply: process 16 bytes at a time via `vqtbl1q_u8`.
///
/// Uses the same subtable strategy as SSE4.2: split the 256-byte table into
/// 16 subtables of 16 bytes. For each input chunk, extract high/low nibbles,
/// lookup via `vqtbl1q_u8`, mask-select by high nibble, and OR into result.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn apply_neon(view: &super::ElementWiseView, data: &mut [u8]) {
    use core::arch::aarch64::*;

    let table = view.table();
    let len = data.len();
    let chunks = len / 16;
    let remainder = chunks * 16;

    unsafe {
        let low_mask = vdupq_n_u8(0x0F);

        for chunk in 0..chunks {
            let ptr = data.as_mut_ptr().add(chunk * 16);
            let input = vld1q_u8(ptr);
            let lo = vandq_u8(input, low_mask);
            let hi = vandq_u8(vshrq_n_u8(input, 4), low_mask);
            let mut result = vdupq_n_u8(0);

            for sub in 0..16u8 {
                let base = (sub as usize) * 16;
                let subtable = vld1q_u8(table.as_ptr().add(base));
                let match_val = vdupq_n_u8(sub);
                let mask = vceqq_u8(hi, match_val);
                let shuffled = vqtbl1q_u8(subtable, lo);
                result = vorrq_u8(result, vandq_u8(mask, shuffled));
            }

            vst1q_u8(ptr, result);
        }
    }

    // Scalar remainder
    for byte in &mut data[remainder..] {
        *byte = view.apply(*byte);
    }
}

/// WASM SIMD128 in-place apply: process 16 bytes at a time via `i8x16_swizzle`.
///
/// Uses the same subtable strategy as NEON/SSE4.2: split the 256-byte table
/// into 16 subtables of 16 bytes. For each input chunk, extract high/low nibbles,
/// lookup via `i8x16_swizzle`, mask-select by high nibble, and OR into result.
#[cfg(target_arch = "wasm32")]
pub fn apply_wasm_simd128(view: &super::ElementWiseView, data: &mut [u8]) {
    use core::arch::wasm32::*;

    let table = view.table();
    let len = data.len();
    let chunks = len / 16;
    let remainder = chunks * 16;

    let low_mask = u8x16_splat(0x0F);

    for chunk in 0..chunks {
        let off = chunk * 16;
        // Safety: off..off+16 is within bounds (chunk < chunks = len/16),
        // and pointers are derived from valid slice references.
        unsafe {
            let input = v128_load(data[off..].as_ptr() as *const v128);
            let lo = v128_and(input, low_mask);
            let hi = v128_and(u8x16_shr(input, 4), low_mask);
            let mut result = u8x16_splat(0);

            for sub in 0..16u8 {
                let base = (sub as usize) * 16;
                let subtable = v128_load(table[base..].as_ptr() as *const v128);
                let match_val = u8x16_splat(sub);
                let mask = u8x16_eq(hi, match_val);
                let shuffled = i8x16_swizzle(subtable, lo);
                result = v128_or(result, v128_and(mask, shuffled));
            }

            v128_store(data[off..].as_mut_ptr() as *mut v128, result);
        }
    }

    // Scalar remainder
    for byte in &mut data[remainder..] {
        *byte = view.apply(*byte);
    }
}

/// WASM SIMD128 separate input/output apply.
#[cfg(target_arch = "wasm32")]
pub fn apply_to_wasm_simd128(view: &super::ElementWiseView, input: &[u8], output: &mut [u8]) {
    use core::arch::wasm32::*;

    let table = view.table();
    let len = input.len();
    let chunks = len / 16;
    let remainder = chunks * 16;

    let low_mask = u8x16_splat(0x0F);

    for chunk in 0..chunks {
        let off = chunk * 16;
        // Safety: same bounds guarantee as apply_wasm_simd128.
        unsafe {
            let inv = v128_load(input[off..].as_ptr() as *const v128);
            let lo = v128_and(inv, low_mask);
            let hi = v128_and(u8x16_shr(inv, 4), low_mask);
            let mut result = u8x16_splat(0);

            for sub in 0..16u8 {
                let base = (sub as usize) * 16;
                let subtable = v128_load(table[base..].as_ptr() as *const v128);
                let match_val = u8x16_splat(sub);
                let mask = u8x16_eq(hi, match_val);
                let shuffled = i8x16_swizzle(subtable, lo);
                result = v128_or(result, v128_and(mask, shuffled));
            }

            v128_store(output[off..].as_mut_ptr() as *mut v128, result);
        }
    }

    // Scalar remainder
    for i in remainder..len {
        output[i] = view.apply(input[i]);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "x86_64", target_feature = "sse4.2"),
    ))]
    use super::super::ElementWiseView;
    #[cfg(all(
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "sse4.2")
    ))]
    use std::vec;
    #[cfg(all(
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "sse4.2")
    ))]
    use std::vec::Vec;

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_identity() {
        let id = ElementWiseView::identity();
        let mut data: Vec<u8> = (0..=255).collect();
        let expected: Vec<u8> = (0..=255).collect();
        super::apply_avx2(&id, &mut data);
        assert_eq!(data, expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_increment() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let mut data: Vec<u8> = (0..=255).collect();
        super::apply_avx2(&inc, &mut data);
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(b, (i as u8).wrapping_add(1));
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_with_remainder() {
        let xor = ElementWiseView::new(|x| x ^ 0xAA);
        let mut data: Vec<u8> = (0..50).collect();
        let expected: Vec<u8> = (0..50u8).map(|x| x ^ 0xAA).collect();
        super::apply_avx2(&xor, &mut data);
        assert_eq!(data, expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_apply_to() {
        let mul = ElementWiseView::new(|x| x.wrapping_mul(3));
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0u8; 256];
        super::apply_to_avx2(&mul, &input, &mut output);
        for (i, &b) in output.iter().enumerate() {
            assert_eq!(b, (i as u8).wrapping_mul(3));
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn sse42_identity() {
        let id = ElementWiseView::identity();
        let mut data: Vec<u8> = (0..=255).collect();
        let expected: Vec<u8> = (0..=255).collect();
        super::apply_sse42(&id, &mut data);
        assert_eq!(data, expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn sse42_increment() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let mut data: Vec<u8> = (0..=255).collect();
        super::apply_sse42(&inc, &mut data);
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(b, (i as u8).wrapping_add(1));
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn sse42_with_remainder() {
        let xor = ElementWiseView::new(|x| x ^ 0xAA);
        let mut data: Vec<u8> = (0..25).collect();
        let expected: Vec<u8> = (0..25u8).map(|x| x ^ 0xAA).collect();
        super::apply_sse42(&xor, &mut data);
        assert_eq!(data, expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    fn sse42_apply_to() {
        let mul = ElementWiseView::new(|x| x.wrapping_mul(3));
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0u8; 256];
        super::apply_to_sse42(&mul, &input, &mut output);
        for (i, &b) in output.iter().enumerate() {
            assert_eq!(b, (i as u8).wrapping_mul(3));
        }
    }
}
