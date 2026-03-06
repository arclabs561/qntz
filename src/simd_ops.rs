//! Bit-level / popcount helpers for quantization codes.
//!
//! The name says “simd”, but the implementation is portable and relies on
//! `count_ones` on integer words (which is typically compiled to POPCNT).

use crate::VQuantError;

/// Pack binary codes (0/1 bytes) into a bitfield.
///
/// Each input byte is treated as a boolean: nonzero becomes a set bit.
///
/// # Buffer requirements
///
/// `packed.len()` must be at least `codes.len().div_ceil(8)`.
///
/// # Errors
///
/// Returns [`VQuantError::DimensionMismatch`] if the output buffer is too small.
#[inline]
pub fn pack_binary_fast(codes: &[u8], packed: &mut [u8]) -> crate::Result<()> {
    let required = codes.len().div_ceil(8);
    if packed.len() < required {
        return Err(VQuantError::DimensionMismatch {
            expected: required,
            got: packed.len(),
        });
    }
    let full_bytes = codes.len() / 8;

    for (byte_idx, packed_byte) in packed.iter_mut().enumerate().take(full_bytes) {
        let base = byte_idx * 8;
        let mut byte = 0u8;

        if codes[base] != 0 {
            byte |= 1 << 0;
        }
        if codes[base + 1] != 0 {
            byte |= 1 << 1;
        }
        if codes[base + 2] != 0 {
            byte |= 1 << 2;
        }
        if codes[base + 3] != 0 {
            byte |= 1 << 3;
        }
        if codes[base + 4] != 0 {
            byte |= 1 << 4;
        }
        if codes[base + 5] != 0 {
            byte |= 1 << 5;
        }
        if codes[base + 6] != 0 {
            byte |= 1 << 6;
        }
        if codes[base + 7] != 0 {
            byte |= 1 << 7;
        }

        *packed_byte = byte;
    }

    let remaining = codes.len() % 8;
    if remaining > 0 {
        let base = full_bytes * 8;
        let mut byte = 0u8;
        for i in 0..remaining {
            if codes[base + i] != 0 {
                byte |= 1 << i;
            }
        }
        packed[full_bytes] = byte;
    }

    Ok(())
}

/// Unpack a bitfield back into one byte per bit (0 or 1).
///
/// Inverse of [`pack_binary_fast`]. `dim` is the number of codes to extract.
///
/// # Buffer requirements
///
/// - `packed.len()` must be at least `dim.div_ceil(8)`.
/// - `codes.len()` must be at least `dim`.
///
/// # Errors
///
/// Returns [`VQuantError::DimensionMismatch`] if either buffer is too small.
#[inline]
pub fn unpack_binary_fast(packed: &[u8], codes: &mut [u8], dim: usize) -> crate::Result<()> {
    let required_packed = dim.div_ceil(8);
    if packed.len() < required_packed {
        return Err(VQuantError::DimensionMismatch {
            expected: required_packed,
            got: packed.len(),
        });
    }
    if codes.len() < dim {
        return Err(VQuantError::DimensionMismatch {
            expected: dim,
            got: codes.len(),
        });
    }
    let full_bytes = dim / 8;

    for (byte_idx, &byte) in packed.iter().enumerate().take(full_bytes) {
        let base = byte_idx * 8;

        codes[base] = byte & 1;
        codes[base + 1] = (byte >> 1) & 1;
        codes[base + 2] = (byte >> 2) & 1;
        codes[base + 3] = (byte >> 3) & 1;
        codes[base + 4] = (byte >> 4) & 1;
        codes[base + 5] = (byte >> 5) & 1;
        codes[base + 6] = (byte >> 6) & 1;
        codes[base + 7] = (byte >> 7) & 1;
    }

    let remaining = dim % 8;
    if remaining > 0 && full_bytes < packed.len() {
        let byte = packed[full_bytes];
        let base = full_bytes * 8;
        for i in 0..remaining {
            codes[base + i] = (byte >> i) & 1;
        }
    }

    Ok(())
}

/// Hamming distance between two packed bit-vectors.
///
/// Counts the number of differing bits across `min(a.len(), b.len())` bytes.
/// Trailing bytes in the longer slice are ignored. Both slices may be empty
/// (returns 0).
#[inline]
#[must_use]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0u32;
    let len = a.len().min(b.len());

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        let a_u64 = u64::from_le_bytes([
            a[base],
            a[base + 1],
            a[base + 2],
            a[base + 3],
            a[base + 4],
            a[base + 5],
            a[base + 6],
            a[base + 7],
        ]);
        let b_u64 = u64::from_le_bytes([
            b[base],
            b[base + 1],
            b[base + 2],
            b[base + 3],
            b[base + 4],
            b[base + 5],
            b[base + 6],
            b[base + 7],
        ]);
        dist += (a_u64 ^ b_u64).count_ones();
    }

    for i in (chunks * 8)..len {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

/// Asymmetric inner product: f32 query vs packed 1-bit codes.
///
/// Convention: `bit=1 -> +1`, `bit=0 -> -1`.
///
/// # Buffer requirements
///
/// `codes.len()` must be at least `query.len().div_ceil(8)`.
///
/// # Errors
///
/// Returns [`VQuantError::DimensionMismatch`] if the codes buffer is too small
/// for the query dimension.
#[inline]
pub fn asymmetric_inner_product(query: &[f32], codes: &[u8]) -> crate::Result<f32> {
    let dim = query.len();
    let required = dim.div_ceil(8);
    if codes.len() < required {
        return Err(VQuantError::DimensionMismatch {
            expected: required,
            got: codes.len(),
        });
    }
    let mut sum = 0.0f32;

    let full_bytes = dim / 8;
    for (byte_idx, &byte) in codes.iter().enumerate().take(full_bytes) {
        let base = byte_idx * 8;

        sum += if byte & 1 != 0 {
            query[base]
        } else {
            -query[base]
        };
        sum += if byte & 2 != 0 {
            query[base + 1]
        } else {
            -query[base + 1]
        };
        sum += if byte & 4 != 0 {
            query[base + 2]
        } else {
            -query[base + 2]
        };
        sum += if byte & 8 != 0 {
            query[base + 3]
        } else {
            -query[base + 3]
        };
        sum += if byte & 16 != 0 {
            query[base + 4]
        } else {
            -query[base + 4]
        };
        sum += if byte & 32 != 0 {
            query[base + 5]
        } else {
            -query[base + 5]
        };
        sum += if byte & 64 != 0 {
            query[base + 6]
        } else {
            -query[base + 6]
        };
        sum += if byte & 128 != 0 {
            query[base + 7]
        } else {
            -query[base + 7]
        };
    }

    let remaining = dim % 8;
    if remaining > 0 && full_bytes < codes.len() {
        let byte = codes[full_bytes];
        let base = full_bytes * 8;
        for i in 0..remaining {
            let sign = if (byte >> i) & 1 != 0 { 1.0 } else { -1.0 };
            sum += sign * query[base + i];
        }
    }

    Ok(sum)
}

/// Asymmetric L2 distance squared: `||q - b||^2 = ||q||^2 + D - 2<q, b>`
/// where `b in {-1,+1}^D`.
///
/// # Errors
///
/// Returns [`VQuantError::DimensionMismatch`] if the codes buffer is too small
/// for the query dimension (see [`asymmetric_inner_product`]).
#[inline]
pub fn asymmetric_l2_squared(query: &[f32], codes: &[u8]) -> crate::Result<f32> {
    let dim = query.len();
    let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    let ip = asymmetric_inner_product(query, codes)?;
    Ok(query_norm_sq + dim as f32 - 2.0 * ip)
}

/// Batch Hamming distances from `query` to each element in `codes`.
///
/// Returns one `u32` per element, using `min(query.len(), code.len())` bytes
/// for each comparison.
#[inline]
#[must_use]
pub fn batch_hamming_distances(query: &[u8], codes: &[&[u8]]) -> Vec<u32> {
    codes.iter().map(|c| hamming_distance(query, c)).collect()
}

/// Batch asymmetric L2 squared distances from `query` to each element in `codes`.
///
/// # Errors
///
/// Returns the first [`VQuantError::DimensionMismatch`] encountered if any
/// codes buffer is too small for the query dimension.
#[inline]
pub fn batch_asymmetric_l2(query: &[f32], codes: &[&[u8]]) -> crate::Result<Vec<f32>> {
    codes
        .iter()
        .map(|c| asymmetric_l2_squared(query, c))
        .collect()
}

/// Pack extended codes (`ex_bits` per element) into a bitfield.
///
/// Each code is masked to its low `ex_bits` bits before packing.
/// No-op when `ex_bits == 0`.
///
/// # Buffer requirements
///
/// `packed.len()` must be at least `(codes.len() * ex_bits).div_ceil(8)`.
/// Bits beyond the buffer length are silently dropped.
#[inline]
pub fn pack_extended_interleaved(codes: &[u16], packed: &mut [u8], ex_bits: usize) {
    if ex_bits == 0 {
        return;
    }

    let mut bit_pos = 0;
    for &code in codes {
        let val = code & ((1 << ex_bits) - 1);
        for b in 0..ex_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if byte_idx < packed.len() && (val >> b) & 1 != 0 {
                packed[byte_idx] |= 1 << bit_idx;
            }
            bit_pos += 1;
        }
    }
}

/// Unpack extended codes from a bitfield.
///
/// Inverse of [`pack_extended_interleaved`]. Extracts `dim` codes of `ex_bits`
/// each from `packed`. When `ex_bits == 0`, fills output codes with zeros.
///
/// # Buffer requirements
///
/// - `packed.len()` must be at least `(dim * ex_bits).div_ceil(8)`.
/// - `codes.len()` must be at least `dim`.
///
/// Bits beyond the packed buffer are read as zero.
#[inline]
pub fn unpack_extended_interleaved(packed: &[u8], codes: &mut [u16], dim: usize, ex_bits: usize) {
    if ex_bits == 0 {
        codes.iter_mut().for_each(|c| *c = 0);
        return;
    }

    let mut bit_pos = 0;
    for code in codes.iter_mut().take(dim) {
        let mut val = 0u16;
        for b in 0..ex_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 != 0 {
                val |= 1 << b;
            }
            bit_pos += 1;
        }
        *code = val;
    }
}

/// Inner product with multi-bit quantized codes.
///
/// Each code is centered at `(2^bits - 1) / 2`, so code 0 maps to `-center`
/// and the max code maps to `+center`.
#[inline]
#[must_use]
pub fn multibit_inner_product(query: &[f32], codes: &[u16], total_bits: usize) -> f32 {
    let center = ((1 << total_bits) as f32 - 1.0) / 2.0;
    query
        .iter()
        .zip(codes.iter())
        .map(|(q, &c)| q * (c as f32 - center))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_binary_roundtrip() {
        let codes = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1];
        let mut packed = vec![0u8; 2];
        pack_binary_fast(&codes, &mut packed).unwrap();

        let mut unpacked = vec![0u8; 16];
        unpack_binary_fast(&packed, &mut unpacked, 16).unwrap();

        assert_eq!(codes, unpacked);
    }

    #[test]
    fn hamming_distance_basic() {
        let a = vec![0b11111111];
        let b = vec![0b00000000];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_asymmetric_inner_product() {
        let query = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let codes = vec![0b11111111];
        let ip = asymmetric_inner_product(&query, &codes).unwrap();
        assert!((ip - 8.0).abs() < 1e-6);

        let codes_neg = vec![0b00000000];
        let ip_neg = asymmetric_inner_product(&query, &codes_neg).unwrap();
        assert!((ip_neg - (-8.0)).abs() < 1e-6);
    }

    #[test]
    fn test_multibit_inner_product() {
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let codes: Vec<u16> = vec![15, 15, 0, 0];
        let ip = multibit_inner_product(&query, &codes, 4);
        assert!((ip - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_pack_unpack_extended_interleaved() {
        let codes: Vec<u16> = vec![3, 1, 7, 0, 5, 2, 6, 4];
        let ex_bits = 3;
        let packed_size = (codes.len() * ex_bits).div_ceil(8);
        let mut packed = vec![0u8; packed_size];

        pack_extended_interleaved(&codes, &mut packed, ex_bits);

        let mut unpacked = vec![0u16; codes.len()];
        unpack_extended_interleaved(&packed, &mut unpacked, codes.len(), ex_bits);

        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_batch_hamming() {
        let query = vec![0b11111111];
        let codes: Vec<&[u8]> = vec![&[0b11111111], &[0b11110000], &[0b00000000]];

        let distances = batch_hamming_distances(&query, &codes);
        assert_eq!(distances, vec![0, 4, 8]);
    }

    #[test]
    fn hamming_empty() {
        assert_eq!(hamming_distance(&[], &[]), 0);
    }

    #[test]
    fn hamming_identical() {
        let a = vec![0xABu8; 100];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_opposite() {
        let a = vec![0x00u8; 1];
        let b = vec![0xFFu8; 1];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn asymmetric_ip_sign_convention() {
        // Verify that bit=1 -> +1, bit=0 -> -1
        let query = vec![1.0f32; 8]; // all positive
        let codes_all_ones = vec![0xFFu8]; // all +1
        let codes_all_zeros = vec![0x00u8]; // all -1
        let ip_ones = asymmetric_inner_product(&query, &codes_all_ones).unwrap();
        let ip_zeros = asymmetric_inner_product(&query, &codes_all_zeros).unwrap();
        assert!(ip_ones > 0.0, "all +1 codes should give positive IP");
        assert!(ip_zeros < 0.0, "all -1 codes should give negative IP");
        assert_eq!(ip_ones, -ip_zeros, "should be symmetric around 0");
    }

    #[test]
    fn multibit_ip_centering() {
        // With total_bits=1, codes in {0,1} centered at 0.5
        // So code=0 -> -0.5, code=1 -> +0.5
        let query = vec![2.0f32; 4];
        let codes = vec![1u16, 0, 1, 0]; // +0.5, -0.5, +0.5, -0.5
        let ip = multibit_inner_product(&query, &codes, 1);
        // Expected: 2*0.5 + 2*(-0.5) + 2*0.5 + 2*(-0.5) = 0
        assert!((ip - 0.0).abs() < 1e-6, "multibit IP centering: got {}", ip);
    }

    // ---- error case tests ----

    #[test]
    fn pack_binary_undersized_output() {
        let codes = vec![1u8; 16]; // needs 2 bytes
        let mut packed = vec![0u8; 1]; // only 1 byte
        assert!(pack_binary_fast(&codes, &mut packed).is_err());
    }

    #[test]
    fn unpack_binary_undersized_packed() {
        let packed = vec![0u8; 1]; // only 8 bits
        let mut codes = vec![0u8; 16];
        assert!(unpack_binary_fast(&packed, &mut codes, 16).is_err());
    }

    #[test]
    fn unpack_binary_undersized_output() {
        let packed = vec![0u8; 2];
        let mut codes = vec![0u8; 8]; // needs 16
        assert!(unpack_binary_fast(&packed, &mut codes, 16).is_err());
    }

    #[test]
    fn asymmetric_ip_undersized_codes() {
        let query = vec![1.0f32; 16]; // needs 2 bytes of codes
        let codes = vec![0xFFu8; 1]; // only 1 byte
        assert!(asymmetric_inner_product(&query, &codes).is_err());
    }

    #[test]
    fn asymmetric_l2_undersized_codes() {
        let query = vec![1.0f32; 16];
        let codes = vec![0xFFu8; 1];
        assert!(asymmetric_l2_squared(&query, &codes).is_err());
    }

    #[test]
    fn pack_binary_empty() {
        let codes: Vec<u8> = vec![];
        let mut packed: Vec<u8> = vec![];
        pack_binary_fast(&codes, &mut packed).unwrap();
    }

    #[test]
    fn unpack_binary_empty() {
        let packed: Vec<u8> = vec![];
        let mut codes: Vec<u8> = vec![];
        unpack_binary_fast(&packed, &mut codes, 0).unwrap();
    }
}
