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
    // With the `simd` feature this dispatches to innr's runtime
    // AVX-512/AVX2/NEON popcount; the portable body below is the fallback.
    // innr::hamming_distance asserts equal length, so slice to the shorter
    // first to keep the overlapping-bytes contract documented above.
    #[cfg(feature = "simd")]
    let out = {
        let len = a.len().min(b.len());
        innr::hamming_distance(&a[..len], &b[..len])
    };
    #[cfg(not(feature = "simd"))]
    let out = hamming_distance_portable(a, b);
    out
}

#[inline]
#[cfg(not(feature = "simd"))]
fn hamming_distance_portable(a: &[u8], b: &[u8]) -> u32 {
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
    let mut distances = Vec::with_capacity(codes.len());
    batch_hamming_distances_into(query, codes, &mut distances);
    distances
}

/// Batch Hamming distances from `query` to each element in `codes`, reusing
/// `distances` across repeated scans.
///
/// Clears `distances` before writing one `u32` per element.
#[inline]
pub fn batch_hamming_distances_into(query: &[u8], codes: &[&[u8]], distances: &mut Vec<u32>) {
    distances.clear();
    distances.reserve(codes.len());
    distances.extend(codes.iter().map(|c| hamming_distance(query, c)));
}

/// Batch asymmetric L2 squared distances from `query` to each element in `codes`.
///
/// # Errors
///
/// Returns the first [`VQuantError::DimensionMismatch`] encountered if any
/// codes buffer is too small for the query dimension.
#[inline]
pub fn batch_asymmetric_l2(query: &[f32], codes: &[&[u8]]) -> crate::Result<Vec<f32>> {
    let mut distances = Vec::with_capacity(codes.len());
    batch_asymmetric_l2_into(query, codes, &mut distances)?;
    Ok(distances)
}

/// Batch asymmetric L2 squared distances from `query` to each element in
/// `codes`, reusing `distances` across repeated scans.
///
/// Clears `distances` before writing one `f32` per element.
///
/// # Errors
///
/// Returns the first [`VQuantError::DimensionMismatch`] encountered if any
/// codes buffer is too small for the query dimension.
#[inline]
pub fn batch_asymmetric_l2_into(
    query: &[f32],
    codes: &[&[u8]],
    distances: &mut Vec<f32>,
) -> crate::Result<()> {
    let required = query.len().div_ceil(8);
    for code in codes {
        if code.len() < required {
            return Err(VQuantError::DimensionMismatch {
                expected: required,
                got: code.len(),
            });
        }
    }

    distances.clear();
    distances.reserve(codes.len());
    for code in codes {
        distances.push(asymmetric_l2_squared(query, code)?);
    }
    Ok(())
}

/// Pack extended codes (`ex_bits` per element) into a bitfield.
///
/// Each code is masked to its low `ex_bits` bits before packing.
/// No-op when `ex_bits == 0`.
///
/// # Buffer requirements
///
/// `packed.len()` must be at least `(codes.len() * ex_bits).div_ceil(8)`.
/// Invalid inputs leave `packed` unchanged. For validation, prefer
/// [`try_pack_extended_interleaved`].
#[inline]
pub fn pack_extended_interleaved(codes: &[u16], packed: &mut [u8], ex_bits: usize) {
    let _ = try_pack_extended_interleaved(codes, packed, ex_bits);
}

/// Checked form of [`pack_extended_interleaved`].
///
/// The bytes occupied by the encoded values are overwritten, so a destination
/// buffer can be reused without retaining bits from an earlier encoding.
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] when `ex_bits > 16`, or
/// [`VQuantError::DimensionMismatch`] when `packed` is too small.
#[inline]
pub fn try_pack_extended_interleaved(
    codes: &[u16],
    packed: &mut [u8],
    ex_bits: usize,
) -> crate::Result<()> {
    validate_extended_bits(ex_bits)?;
    let required = required_extended_bytes(codes.len(), ex_bits)?;
    if packed.len() < required {
        return Err(VQuantError::DimensionMismatch {
            expected: required,
            got: packed.len(),
        });
    }
    packed[..required].fill(0);
    if ex_bits == 0 {
        return Ok(());
    }

    let mask = extended_mask(ex_bits);
    let mut bit_pos = 0;
    for &code in codes {
        let val = code & mask;
        for b in 0..ex_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if (val >> b) & 1 != 0 {
                packed[byte_idx] |= 1 << bit_idx;
            }
            bit_pos += 1;
        }
    }
    Ok(())
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
/// Invalid inputs leave `codes` unchanged. For validation, prefer
/// [`try_unpack_extended_interleaved`].
#[inline]
pub fn unpack_extended_interleaved(packed: &[u8], codes: &mut [u16], dim: usize, ex_bits: usize) {
    let _ = try_unpack_extended_interleaved(packed, codes, dim, ex_bits);
}

/// Checked form of [`unpack_extended_interleaved`].
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] when `ex_bits > 16`, or
/// [`VQuantError::DimensionMismatch`] when either buffer is too small.
#[inline]
pub fn try_unpack_extended_interleaved(
    packed: &[u8],
    codes: &mut [u16],
    dim: usize,
    ex_bits: usize,
) -> crate::Result<()> {
    validate_extended_bits(ex_bits)?;
    if codes.len() < dim {
        return Err(VQuantError::DimensionMismatch {
            expected: dim,
            got: codes.len(),
        });
    }
    let required = required_extended_bytes(dim, ex_bits)?;
    if packed.len() < required {
        return Err(VQuantError::DimensionMismatch {
            expected: required,
            got: packed.len(),
        });
    }
    if ex_bits == 0 {
        codes[..dim].fill(0);
        return Ok(());
    }

    let mut bit_pos = 0;
    for code in codes.iter_mut().take(dim) {
        let mut val = 0u16;
        for b in 0..ex_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if (packed[byte_idx] >> bit_idx) & 1 != 0 {
                val |= 1 << b;
            }
            bit_pos += 1;
        }
        *code = val;
    }
    Ok(())
}

fn validate_extended_bits(ex_bits: usize) -> crate::Result<()> {
    if ex_bits > u16::BITS as usize {
        return Err(VQuantError::InvalidConfig {
            field: "ex_bits",
            reason: "ex_bits must be in 0..=16 for u16 codes",
        });
    }
    Ok(())
}

fn extended_mask(ex_bits: usize) -> u16 {
    if ex_bits == u16::BITS as usize {
        u16::MAX
    } else {
        (1u16 << ex_bits) - 1
    }
}

fn required_extended_bytes(dim: usize, ex_bits: usize) -> crate::Result<usize> {
    dim.checked_mul(ex_bits)
        .map(|bits| bits.div_ceil(8))
        .ok_or(VQuantError::InvalidConfig {
            field: "dim",
            reason: "encoded bit length overflows usize",
        })
}

/// Inner product with multi-bit quantized codes.
///
/// Each code is centered at `(2^bits - 1) / 2`, so code 0 maps to `-center`
/// and the max code maps to `+center`.
/// Returns NaN for an invalid bit width; use [`try_multibit_inner_product`] to
/// handle invalid configuration explicitly.
#[inline]
#[must_use]
pub fn multibit_inner_product(query: &[f32], codes: &[u16], total_bits: usize) -> f32 {
    try_multibit_inner_product(query, codes, total_bits).unwrap_or(f32::NAN)
}

/// Checked form of [`multibit_inner_product`].
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] unless `total_bits` is in `1..=16`.
#[inline]
pub fn try_multibit_inner_product(
    query: &[f32],
    codes: &[u16],
    total_bits: usize,
) -> crate::Result<f32> {
    if !(1..=u16::BITS as usize).contains(&total_bits) {
        return Err(VQuantError::InvalidConfig {
            field: "total_bits",
            reason: "total_bits must be in 1..=16 for u16 codes",
        });
    }
    let center = ((1u32 << total_bits) as f32 - 1.0) / 2.0;
    Ok(query
        .iter()
        .zip(codes.iter())
        .map(|(q, &c)| q * (c as f32 - center))
        .sum())
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
    fn checked_extended_roundtrip_matches_independent_bitstream_oracle() {
        for ex_bits in 0..=16 {
            let mask = extended_mask(ex_bits);
            let codes: Vec<u16> = (0..73)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => mask,
                    2 => mask / 2,
                    _ => (i as u16).wrapping_mul(40503) & mask,
                })
                .collect();
            let mut packed = vec![0xa5; required_extended_bytes(codes.len(), ex_bits).unwrap()];
            try_pack_extended_interleaved(&codes, &mut packed, ex_bits).unwrap();

            for (i, &expected) in codes.iter().enumerate() {
                let mut oracle = 0u16;
                for bit in 0..ex_bits {
                    let absolute = i * ex_bits + bit;
                    let set = (packed[absolute / 8] >> (absolute % 8)) & 1;
                    oracle |= u16::from(set) << bit;
                }
                assert_eq!(
                    oracle, expected,
                    "independent decode failed at width {ex_bits}"
                );
            }

            let mut decoded = vec![u16::MAX; codes.len()];
            try_unpack_extended_interleaved(&packed, &mut decoded, codes.len(), ex_bits).unwrap();
            assert_eq!(decoded, codes);
        }
    }

    #[test]
    fn checked_extended_pack_overwrites_reused_destination() {
        let mut packed = [u8::MAX; 2];
        try_pack_extended_interleaved(&[0, 0, 0], &mut packed, 3).unwrap();
        assert_eq!(packed, [0, 0]);

        try_pack_extended_interleaved(&[7, 1, 4], &mut packed, 3).unwrap();
        let mut decoded = [0u16; 3];
        try_unpack_extended_interleaved(&packed, &mut decoded, 3, 3).unwrap();
        assert_eq!(decoded, [7, 1, 4]);
    }

    #[test]
    fn checked_extended_operations_reject_invalid_inputs() {
        let mut byte = [0u8; 1];
        let mut code = [0u16; 1];
        assert!(try_pack_extended_interleaved(&[1], &mut [], 1).is_err());
        assert!(try_unpack_extended_interleaved(&[], &mut code, 1, 1).is_err());
        assert!(try_unpack_extended_interleaved(&byte, &mut [], 1, 1).is_err());
        assert!(try_pack_extended_interleaved(&[1], &mut byte, 17).is_err());
        assert!(try_unpack_extended_interleaved(&byte, &mut code, 1, 17).is_err());
        assert!(required_extended_bytes(usize::MAX, 2).is_err());
        assert!(try_multibit_inner_product(&[1.0], &[1], 0).is_err());
        assert!(try_multibit_inner_product(&[1.0], &[1], 17).is_err());
    }

    #[test]
    fn test_batch_hamming() {
        let query = vec![0b11111111];
        let codes: Vec<&[u8]> = vec![&[0b11111111], &[0b11110000], &[0b00000000]];

        let distances = batch_hamming_distances(&query, &codes);
        assert_eq!(distances, vec![0, 4, 8]);

        let mut into = Vec::with_capacity(8);
        let capacity = into.capacity();
        batch_hamming_distances_into(&query, &codes, &mut into);
        assert_eq!(into, distances);
        assert!(
            into.capacity() >= capacity,
            "caller-provided allocation should be reused"
        );
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
    fn batch_asymmetric_l2_into_matches_allocating() {
        let query = vec![1.0f32; 8];
        let a = vec![0b11111111];
        let b = vec![0b00000000];
        let codes: Vec<&[u8]> = vec![&a, &b];

        let distances = batch_asymmetric_l2(&query, &codes).unwrap();
        let mut into = Vec::with_capacity(8);
        let capacity = into.capacity();
        batch_asymmetric_l2_into(&query, &codes, &mut into).unwrap();

        assert_eq!(into, distances);
        assert!(
            into.capacity() >= capacity,
            "caller-provided allocation should be reused"
        );
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
