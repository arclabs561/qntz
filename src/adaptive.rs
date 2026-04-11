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

    /// Quantize multiple vectors into a packed batch representation.
    ///
    /// All vectors must have the same dimensionality. Returns an error if
    /// dimensions are inconsistent, or `Ok` with a [`PackedBatch`] that stores
    /// codes contiguously for cache-friendly scanning.
    pub fn quantize_packed(&self, vectors: &[impl AsRef<[f32]>]) -> Result<PackedBatch> {
        if vectors.is_empty() {
            return Ok(PackedBatch {
                codes: vec![],
                scales: vec![],
                offsets: vec![],
                dim: 0,
                bits: self.bits,
            });
        }
        let dim = vectors[0].as_ref().len();
        for (_i, v) in vectors.iter().enumerate().skip(1) {
            if v.as_ref().len() != dim {
                return Err(VQuantError::DimensionMismatch {
                    expected: dim,
                    got: v.as_ref().len(),
                });
            }
        }

        let n = vectors.len();
        let mut codes = Vec::with_capacity(n * dim);
        let mut scales = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);

        for v in vectors {
            let q = self.quantize(v.as_ref());
            codes.extend_from_slice(&q.codes);
            scales.push(q.scale);
            offsets.push(q.offset);
        }

        Ok(PackedBatch {
            codes,
            scales,
            offsets,
            dim,
            bits: self.bits,
        })
    }

    /// Build a lookup table for fast asymmetric distance computation.
    ///
    /// For a given query vector and bit width, precomputes `(query[d] - recon)^2`
    /// for every possible code value at each dimension. This avoids repeated
    /// multiply-add in the inner loop when scanning many quantized vectors that
    /// share the same scale/offset... but since adaptive quantization uses
    /// per-vector scale/offset, the table must be rebuilt per document vector.
    ///
    /// For scanning a [`PackedBatch`], use [`PackedBatch::asymmetric_distances`]
    /// which handles this internally.
    pub fn build_distance_table(query: &[f32], quantized: &AdaptiveQuantized) -> Vec<f32> {
        let num_codes = 1usize << quantized.bits;
        let dim = query.len();
        let max_code = (num_codes - 1) as f32;
        let step = if quantized.scale == 0.0 {
            0.0
        } else {
            quantized.scale / max_code
        };
        let offset = quantized.offset;

        let mut table = Vec::with_capacity(dim * num_codes);
        for &q in query.iter().take(dim) {
            for c in 0..num_codes {
                let recon = (c as f32) * step + offset;
                let diff = q - recon;
                table.push(diff * diff);
            }
        }
        table
    }

    /// Asymmetric L2 squared distance using a precomputed lookup table.
    ///
    /// The table must have been built for the same query and the same
    /// scale/offset as the quantized vector. This is a building block for
    /// batch scanning; for single-vector distance, [`AdaptiveQuantizer::asymmetric_distance`] is
    /// simpler.
    pub fn distance_from_table(table: &[f32], quantized: &AdaptiveQuantized) -> f32 {
        let num_codes = 1usize << quantized.bits;
        quantized
            .codes
            .iter()
            .enumerate()
            .map(|(d, &c)| {
                // table layout: [dim0_code0, dim0_code1, ..., dim1_code0, ...]
                table[d * num_codes + c as usize]
            })
            .sum()
    }
}

/// Packed representation of multiple quantized vectors.
///
/// Codes are stored contiguously in row-major order: vector 0's codes first,
/// then vector 1's, etc. Each vector has its own scale and offset.
#[derive(Clone, Debug)]
pub struct PackedBatch {
    /// Flat code array: `codes[i * dim .. (i+1) * dim]` for vector `i`.
    pub codes: Vec<u8>,
    /// Per-vector scales.
    pub scales: Vec<f32>,
    /// Per-vector offsets.
    pub offsets: Vec<f32>,
    /// Dimensionality of each vector.
    pub dim: usize,
    /// Bits per code.
    pub bits: u8,
}

impl PackedBatch {
    /// Number of vectors in the batch.
    pub fn len(&self) -> usize {
        self.scales.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Compute asymmetric L2 squared distances from a query to all vectors.
    ///
    /// Returns one distance per vector. The query must have the same
    /// dimensionality as the stored vectors.
    pub fn asymmetric_distances(&self, query: &[f32]) -> Result<Vec<f32>> {
        if !self.is_empty() && query.len() != self.dim {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let n = self.len();
        let dim = self.dim;
        let num_codes = 1usize << self.bits;
        let mut distances = Vec::with_capacity(n);

        for i in 0..n {
            let scale = self.scales[i];
            let offset = self.offsets[i];
            let max_code = (num_codes - 1) as f32;
            let step = if scale == 0.0 { 0.0 } else { scale / max_code };

            let codes_start = i * dim;
            let mut dist = 0.0f32;
            for (d, &q) in query.iter().enumerate().take(dim) {
                let c = self.codes[codes_start + d];
                let recon = (c as f32) * step + offset;
                let diff = q - recon;
                dist += diff * diff;
            }
            distances.push(dist);
        }

        Ok(distances)
    }

    /// Dequantize a single vector from the batch by index.
    ///
    /// Returns `None` if `index >= len()`.
    pub fn dequantize(&self, index: usize) -> Option<Vec<f32>> {
        if index >= self.len() {
            return None;
        }
        let dim = self.dim;
        let max_code = ((1u32 << self.bits) - 1) as f32;
        let scale = self.scales[index];
        let offset = self.offsets[index];
        let step = if scale == 0.0 { 0.0 } else { scale / max_code };

        let codes_start = index * dim;
        let vec = (0..dim)
            .map(|d| (self.codes[codes_start + d] as f32) * step + offset)
            .collect();
        Some(vec)
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
    fn test_packed_batch_roundtrip() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|j| (0..32).map(|i| (i as f32 + j as f32) * 0.1 - 1.0).collect())
            .collect();

        let packed = quantizer.quantize_packed(&vectors).unwrap();
        assert_eq!(packed.len(), 10);
        assert_eq!(packed.codes.len(), 10 * 32);

        for (idx, orig) in vectors.iter().enumerate() {
            let recon = packed.dequantize(idx).unwrap();
            let err_sq: f32 = orig
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let norm_sq: f32 = orig.iter().map(|x| x * x).sum();
            assert!(
                err_sq < 0.005 * norm_sq + 1e-8,
                "packed roundtrip error too high at vector {idx}"
            );
        }
    }

    #[test]
    fn test_packed_batch_dimension_mismatch() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert!(quantizer.quantize_packed(&vectors).is_err());
    }

    #[test]
    fn test_packed_batch_empty() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let empty: Vec<Vec<f32>> = vec![];
        let packed = quantizer.quantize_packed(&empty).unwrap();
        assert!(packed.is_empty());
        assert_eq!(packed.len(), 0);
    }

    #[test]
    fn test_packed_asymmetric_distances() {
        let quantizer = AdaptiveQuantizer::new(8).unwrap();
        let query: Vec<f32> = (0..32).map(|i| (i as f32) * 0.07 - 1.0).collect();
        let docs: Vec<Vec<f32>> = (0..5)
            .map(|j| (0..32).map(|i| (i as f32 + j as f32) * 0.1 - 0.5).collect())
            .collect();

        let packed = quantizer.quantize_packed(&docs).unwrap();
        let dists = packed.asymmetric_distances(&query).unwrap();
        assert_eq!(dists.len(), 5);

        // Compare against individual asymmetric_distance calls.
        for (j, doc) in docs.iter().enumerate() {
            let single_q = quantizer.quantize(doc);
            let single_dist = AdaptiveQuantizer::asymmetric_distance(&query, &single_q);
            let rel_diff = (dists[j] - single_dist).abs() / (single_dist + 1e-12);
            assert!(
                rel_diff < 1e-5,
                "packed vs single distance mismatch at vector {j}: {} vs {}",
                dists[j],
                single_dist
            );
        }
    }

    #[test]
    fn test_packed_asymmetric_distance_dimension_mismatch() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let docs = vec![vec![1.0, 2.0, 3.0]];
        let packed = quantizer.quantize_packed(&docs).unwrap();
        let wrong_query = vec![1.0, 2.0];
        assert!(packed.asymmetric_distances(&wrong_query).is_err());
    }

    #[test]
    fn test_distance_table_matches_direct() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let query: Vec<f32> = (0..16).map(|i| (i as f32) * 0.2 - 1.5).collect();
        let doc: Vec<f32> = (0..16).map(|i| (i as f32) * 0.15 - 1.0).collect();

        let quantized = quantizer.quantize(&doc);
        let direct = AdaptiveQuantizer::asymmetric_distance(&query, &quantized);
        let table = AdaptiveQuantizer::build_distance_table(&query, &quantized);
        let via_table = AdaptiveQuantizer::distance_from_table(&table, &quantized);

        assert!(
            (direct - via_table).abs() < 1e-6,
            "table distance {} != direct distance {}",
            via_table,
            direct
        );
    }

    #[test]
    fn test_dequantize_out_of_bounds() {
        let quantizer = AdaptiveQuantizer::new(4).unwrap();
        let docs = vec![vec![1.0, 2.0]];
        let packed = quantizer.quantize_packed(&docs).unwrap();
        assert!(packed.dequantize(0).is_some());
        assert!(packed.dequantize(1).is_none());
    }

    #[test]
    fn test_asymmetric_distance_accuracy_by_bits() {
        // Higher bits should give distance closer to true distance.
        let query: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let doc: Vec<f32> = (0..64).map(|i| (i as f32 * 0.12) - 2.5).collect();
        let true_dist: f32 = query
            .iter()
            .zip(doc.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        let mut prev_err = f32::INFINITY;
        for bits in [2, 4, 8] {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let quantized = quantizer.quantize(&doc);
            let approx = AdaptiveQuantizer::asymmetric_distance(&query, &quantized);
            let err = (approx - true_dist).abs();
            assert!(
                err < prev_err,
                "{bits}-bit error {err} >= previous {prev_err}"
            );
            prev_err = err;
        }
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

        #[test]
        fn packed_matches_individual(
            vectors in proptest::collection::vec(arb_vector(32), 2..10),
            bits in 1u8..=8,
        ) {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let individual: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();
            let packed = quantizer.quantize_packed(&vectors).unwrap();

            prop_assert_eq!(packed.len(), individual.len());

            let query: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
            let packed_dists = packed.asymmetric_distances(&query).unwrap();

            for (i, ind) in individual.iter().enumerate() {
                let ind_dist = AdaptiveQuantizer::asymmetric_distance(&query, ind);
                let diff = (packed_dists[i] - ind_dist).abs();
                prop_assert!(
                    diff < 1e-4,
                    "vector {i}: packed dist {} vs individual dist {}, diff {diff}",
                    packed_dists[i],
                    ind_dist
                );
            }
        }

        #[test]
        fn distance_table_matches_direct(
            query in arb_vector(32),
            doc in arb_vector(32),
            bits in 1u8..=8,
        ) {
            let quantizer = AdaptiveQuantizer::new(bits).unwrap();
            let quantized = quantizer.quantize(&doc);
            let direct = AdaptiveQuantizer::asymmetric_distance(&query, &quantized);
            let table = AdaptiveQuantizer::build_distance_table(&query, &quantized);
            let via_table = AdaptiveQuantizer::distance_from_table(&table, &quantized);
            let diff = (direct - via_table).abs();
            prop_assert!(
                diff < 1e-4,
                "table dist {via_table} != direct dist {direct}, diff {diff}"
            );
        }
    }
}
