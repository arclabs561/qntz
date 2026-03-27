//! Per-vector adaptive scalar quantization.
//!
//! Each vector gets its own scale and offset derived from its min/max range,
//! rather than sharing a global codebook. This preserves per-vector dynamic
//! range and yields lower reconstruction error than global uniform quantization
//! at the same bit width.
//!
//! Supports asymmetric distance computation (exact query vs quantized document),
//! which avoids quantizing the query and gives tighter distance approximations.
//!
//! # References
//!
//! - Tepper & Willke, "Individualized Non-uniform Quantization for Vector Search" (2025)
//!
//! # Example
//!
//! ```rust
//! use qntz::adaptive::{AdaptiveQuantizer, AdaptiveQuantized};
//!
//! let quantizer = AdaptiveQuantizer::new(8).unwrap();
//! let vector = vec![0.1, -0.5, 1.2, -0.3, 0.7];
//!
//! let quantized = quantizer.quantize(&vector);
//! let reconstructed = AdaptiveQuantizer::dequantize(&quantized);
//!
//! // 8-bit roundtrip error is small relative to vector norm
//! let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
//! let err_sq: f32 = vector.iter().zip(reconstructed.iter())
//!     .map(|(a, b)| (a - b) * (a - b)).sum();
//! assert!(err_sq < 0.005 * norm_sq);
//! ```

use crate::{Result, VQuantError};

/// Per-vector adaptive scalar quantizer.
///
/// Quantizes each element of a vector into `bits`-bit codes using the vector's
/// own min/max range as scale and offset.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveQuantizer {
    bits: u8,
    max_code: f32, // (2^bits - 1) as f32, cached
}

/// A single quantized vector with per-vector scale and offset.
#[derive(Clone, Debug)]
pub struct AdaptiveQuantized {
    /// Quantized codes, one per dimension. Values in `[0, 2^bits - 1]`.
    pub codes: Vec<u8>,
    /// Per-vector scale: `max(vector) - min(vector)`.
    pub scale: f32,
    /// Per-vector offset: `min(vector)`.
    pub offset: f32,
    /// Bits per code.
    pub bits: u8,
}

impl AdaptiveQuantizer {
    /// Create a quantizer for the given bit width.
    ///
    /// `bits` must be in `[1, 8]`.
    pub fn new(bits: u8) -> Result<Self> {
        if bits == 0 || bits > 8 {
            return Err(VQuantError::InvalidConfig {
                field: "bits",
                reason: "bits must be in [1, 8]",
            });
        }
        let max_code = ((1u32 << bits) - 1) as f32;
        Ok(Self { bits, max_code })
    }

    /// Quantize a single vector.
    ///
    /// For a constant vector (all elements equal), scale is 0 and all codes are 0.
    /// For a zero-length vector, returns an empty codes vec.
    pub fn quantize(&self, vector: &[f32]) -> AdaptiveQuantized {
        if vector.is_empty() {
            return AdaptiveQuantized {
                codes: vec![],
                scale: 0.0,
                offset: 0.0,
                bits: self.bits,
            };
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in vector {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let scale = max_val - min_val;
        let offset = min_val;

        let codes = if scale == 0.0 {
            // Constant vector: all codes are 0.
            vec![0u8; vector.len()]
        } else {
            let inv_scale = self.max_code / scale;
            vector
                .iter()
                .map(|&v| {
                    let normalized = (v - offset) * inv_scale;
                    // Clamp to valid range (handles floating-point edge cases).
                    (normalized.round() as u8).min(self.max_code as u8)
                })
                .collect()
        };

        AdaptiveQuantized {
            codes,
            scale,
            offset,
            bits: self.bits,
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(quantized: &AdaptiveQuantized) -> Vec<f32> {
        if quantized.codes.is_empty() {
            return vec![];
        }
        let max_code = ((1u32 << quantized.bits) - 1) as f32;
        if quantized.scale == 0.0 {
            return vec![quantized.offset; quantized.codes.len()];
        }
        let step = quantized.scale / max_code;
        quantized
            .codes
            .iter()
            .map(|&c| (c as f32) * step + quantized.offset)
            .collect()
    }

    /// Asymmetric L2 squared distance: exact query vs quantized document.
    ///
    /// More accurate than symmetric distance (both quantized) because the query
    /// is not quantized.
    pub fn asymmetric_distance(query: &[f32], quantized: &AdaptiveQuantized) -> f32 {
        debug_assert_eq!(query.len(), quantized.codes.len());
        if quantized.codes.is_empty() {
            return 0.0;
        }
        let max_code = ((1u32 << quantized.bits) - 1) as f32;
        let step = if quantized.scale == 0.0 {
            0.0
        } else {
            quantized.scale / max_code
        };
        let offset = quantized.offset;

        query
            .iter()
            .zip(quantized.codes.iter())
            .map(|(&q, &c)| {
                let recon = (c as f32) * step + offset;
                let diff = q - recon;
                diff * diff
            })
            .sum()
    }

    /// Batch quantize multiple vectors.
    pub fn quantize_batch(&self, vectors: &[Vec<f32>]) -> Vec<AdaptiveQuantized> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_8bit() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();

        let quantized = quantizer.quantize(&vector);
        let reconstructed = AdaptiveQuantizer::dequantize(&quantized);

        assert_eq!(vector.len(), reconstructed.len());
        let err_sq: f32 = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        // 8-bit: error < 0.5% of norm squared
        assert!(
            err_sq < 0.005 * norm_sq,
            "8-bit roundtrip error too high: {err_sq} vs norm_sq {norm_sq}"
        );
    }

    #[test]
    fn test_roundtrip_4bit() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();

        let quantized = quantizer.quantize(&vector);
        let reconstructed = AdaptiveQuantizer::dequantize(&quantized);

        assert_eq!(vector.len(), reconstructed.len());
        let err_sq: f32 = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        // 4-bit: error < 5% of norm squared
        assert!(
            err_sq < 0.05 * norm_sq,
            "4-bit roundtrip error too high: {err_sq} vs norm_sq {norm_sq}"
        );
    }

    #[test]
    fn test_scale_offset_correct() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vector = vec![-2.0, 0.0, 1.0, 3.0, 5.0];
        let quantized = quantizer.quantize(&vector);

        assert!(
            (quantized.offset - (-2.0)).abs() < 1e-6,
            "offset should be min"
        );
        assert!(
            (quantized.scale - 7.0).abs() < 1e-6,
            "scale should be max - min"
        );
    }

    #[test]
    fn test_codes_in_range() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let vector: Vec<f32> = (0..100).map(|i| (i as f32) * 0.37 - 20.0).collect();
        let quantized = quantizer.quantize(&vector);
        let max_code = (1u8 << 4) - 1;
        for &c in &quantized.codes {
            assert!(c <= max_code, "code {c} exceeds max {max_code}");
        }
    }

    #[test]
    fn test_zero_vector() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vector = vec![0.0, 0.0, 0.0, 0.0];
        let quantized = quantizer.quantize(&vector);
        let reconstructed = AdaptiveQuantizer::dequantize(&quantized);

        assert_eq!(quantized.scale, 0.0);
        for &r in &reconstructed {
            assert!((r - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_constant_vector() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vector = vec![42.0, 42.0, 42.0];
        let quantized = quantizer.quantize(&vector);
        let reconstructed = AdaptiveQuantizer::dequantize(&quantized);

        assert_eq!(quantized.scale, 0.0);
        assert!((quantized.offset - 42.0).abs() < 1e-6);
        for &r in &reconstructed {
            assert!((r - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_empty_vector() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let quantized = quantizer.quantize(&[]);
        let reconstructed = AdaptiveQuantizer::dequantize(&quantized);
        assert!(quantized.codes.is_empty());
        assert!(reconstructed.is_empty());
    }

    #[test]
    fn test_asymmetric_distance_approximates_l2() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.12) - 2.5).collect();

        let true_dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();

        let quantized_b = quantizer.quantize(&b);
        let approx_dist = AdaptiveQuantizer::asymmetric_distance(&a, &quantized_b);

        // 8-bit asymmetric should be within 5% of true distance
        let rel_err = (approx_dist - true_dist).abs() / true_dist;
        assert!(
            rel_err < 0.05,
            "asymmetric distance relative error too high: {rel_err}"
        );
    }

    #[test]
    fn test_asymmetric_distance_nonneg() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let query = vec![1.0, -1.0, 0.5];
        let doc = vec![0.0, 0.0, 0.0];
        let quantized = quantizer.quantize(&doc);
        let dist = AdaptiveQuantizer::asymmetric_distance(&query, &quantized);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_batch_quantize() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|j| (0..32).map(|i| (i as f32 + j as f32) * 0.1).collect())
            .collect();
        let batch = quantizer.quantize_batch(&vectors);
        assert_eq!(batch.len(), 10);
        for q in &batch {
            assert_eq!(q.codes.len(), 32);
        }
    }

    #[test]
    fn test_invalid_bits() {
        assert!(AdaptiveQuantizer::new(0).is_err());
        assert!(AdaptiveQuantizer::new(9).is_err());
        assert!(AdaptiveQuantizer::new(1).is_ok());
        assert!(AdaptiveQuantizer::new(8).is_ok());
    }

    #[test]
    fn test_more_bits_lower_error() {
        let vector: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05) - 3.0).collect();
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();

        let q4 = AdaptiveQuantizer::new(4).unwrap();
        let q8 = AdaptiveQuantizer::new(8).unwrap();

        let r4 = AdaptiveQuantizer::dequantize(&q4.quantize(&vector));
        let r8 = AdaptiveQuantizer::dequantize(&q8.quantize(&vector));

        let err4: f32 = vector
            .iter()
            .zip(r4.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / norm_sq;
        let err8: f32 = vector
            .iter()
            .zip(r8.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / norm_sq;

        assert!(
            err8 < err4,
            "8-bit should have lower error than 4-bit: {err8} vs {err4}"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_vector(len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, len)
    }

    proptest! {
        #[test]
        fn roundtrip_error_bounded(vector in arb_vector(64), bits in 1u8..=8) {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let quantized = quantizer.quantize(&vector);
            let reconstructed = AdaptiveQuantizer::dequantize(&quantized);

            prop_assert_eq!(vector.len(), reconstructed.len());

            // Theoretical max per-element error: scale / (2^bits - 1) / 2
            // (half a quantization step).
            let max_code = ((1u32 << bits) - 1) as f32;
            let max_elem_err = if quantized.scale == 0.0 {
                0.0
            } else {
                quantized.scale / max_code / 2.0
            };

            for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
                let err = (orig - recon).abs();
                prop_assert!(
                    err <= max_elem_err + 1e-5,
                    "element error {err} exceeds bound {max_elem_err}"
                );
            }
        }

        #[test]
        fn asymmetric_distance_nonneg(
            query in arb_vector(32),
            doc in arb_vector(32),
            bits in 1u8..=8,
        ) {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let quantized = quantizer.quantize(&doc);
            let dist = AdaptiveQuantizer::asymmetric_distance(&query, &quantized);
            prop_assert!(dist >= 0.0, "distance should be non-negative, got {dist}");
        }

        #[test]
        fn codes_in_range(vector in arb_vector(64), bits in 1u8..=8) {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let quantized = quantizer.quantize(&vector);
            let max_code = ((1u32 << bits) - 1) as u8;
            for &c in &quantized.codes {
                prop_assert!(c <= max_code, "code {c} exceeds max {max_code}");
            }
        }
    }
}
