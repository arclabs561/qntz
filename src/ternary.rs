//! Ternary (1.58-bit) quantization.
//!
//! Each dimension is quantized to {-1, 0, +1} and stored as packed 2-bit codes.
//! Configurable thresholds control which values map to zero (the "dead zone"),
//! and adaptive fitting can target a desired sparsity level.
//!
//! # Example
//!
//! ```rust
//! use qntz::ternary::{TernaryQuantizer, TernaryConfig, ternary_hamming};
//!
//! let config = TernaryConfig {
//!     threshold_high: 0.3,
//!     threshold_low: -0.3,
//!     normalize: false,
//!     target_sparsity: None,
//! };
//! let quantizer = TernaryQuantizer::new(4, config);
//!
//! let a = quantizer.quantize(&[0.5, -0.5, 0.1, -0.1]).unwrap();
//! assert_eq!(a.get(0),  1);  // 0.5  > 0.3  -> +1
//! assert_eq!(a.get(1), -1);  // -0.5 < -0.3 -> -1
//! assert_eq!(a.get(2),  0);  // 0.1  in dead zone
//! assert_eq!(a.get(3),  0);  // -0.1 in dead zone
//!
//! let b = quantizer.quantize(&[0.5, 0.5, 0.0, -0.5]).unwrap();
//! assert_eq!(ternary_hamming(&a, &b), Some(2));
//! ```

use crate::VQuantError;

/// Ternary quantized vector.
///
/// Each dimension is stored as 2 bits:
/// - 00 = 0
/// - 01 = +1
/// - 10 = -1
/// - 11 = reserved
#[derive(Clone, Debug)]
pub struct TernaryVector {
    /// Packed 2-bit values (4 values per byte)
    data: Vec<u8>,
    /// Original dimension
    dimension: usize,
    /// Number of +1 values
    positive_count: usize,
    /// Number of -1 values
    negative_count: usize,
    /// Norm of original vector (for asymmetric distance)
    original_norm: f32,
}

impl TernaryVector {
    /// Number of dimensions in the quantized vector.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// L2 norm of the original (pre-normalization) vector.
    ///
    /// Useful for asymmetric scoring conventions where the query is exact.
    #[must_use]
    pub fn original_norm(&self) -> f32 {
        self.original_norm
    }

    /// Get the ternary value at `idx`: returns -1, 0, or +1.
    ///
    /// Returns 0 for out-of-bounds indices.
    #[must_use]
    pub fn get(&self, idx: usize) -> i8 {
        if idx >= self.dimension {
            return 0;
        }
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = (self.data[byte_idx] >> bit_offset) & 0b11;
        match bits {
            0b00 => 0,
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        }
    }

    /// Fraction of zero-valued dimensions (0.0 = fully dense, 1.0 = all zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        let nonzero = self.positive_count + self.negative_count;
        1.0 - (nonzero as f32 / self.dimension as f32)
    }

    /// Number of bytes used for the packed ternary data.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Ternary quantizer configuration.
#[derive(Clone, Debug)]
pub struct TernaryConfig {
    /// Upper threshold: values above this become +1.
    pub threshold_high: f32,
    /// Lower threshold: values below this become -1.
    pub threshold_low: f32,
    /// Whether to L2-normalize input vectors before thresholding.
    pub normalize: bool,
    /// If set, adaptively adjust thresholds to target this fraction of zeros.
    pub target_sparsity: Option<f32>,
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: true,
            target_sparsity: None,
        }
    }
}

/// Quantizer that maps each dimension to {-1, 0, +1}.
pub struct TernaryQuantizer {
    config: TernaryConfig,
    dimension: usize,
    adaptive_thresholds: Option<Vec<(f32, f32)>>,
    mean: Option<Vec<f32>>,
}

impl TernaryQuantizer {
    /// Create a new ternary quantizer with the given dimension and config.
    #[must_use]
    pub fn new(dimension: usize, config: TernaryConfig) -> Self {
        Self {
            config,
            dimension,
            adaptive_thresholds: None,
            mean: None,
        }
    }

    /// Create a quantizer with default config for the given dimension.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, TernaryConfig::default())
    }

    /// Fit adaptive thresholds from training vectors.
    ///
    /// Computes per-dimension mean and, if `target_sparsity` is set, per-dimension
    /// thresholds that achieve the desired sparsity.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> crate::Result<()> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: num_vectors * self.dimension,
                got: vectors.len(),
            });
        }

        let mut mean = vec![0.0f32; self.dimension];
        for i in 0..num_vectors {
            let vec = &vectors[i * self.dimension..(i + 1) * self.dimension];
            for (j, &v) in vec.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= num_vectors as f32;
        }
        self.mean = Some(mean);

        if let Some(target_sparsity) = self.config.target_sparsity {
            let mut thresholds = Vec::with_capacity(self.dimension);

            for d in 0..self.dimension {
                let mut values: Vec<f32> = (0..num_vectors)
                    .map(|i| {
                        let v = vectors[i * self.dimension + d];
                        if let Some(ref m) = self.mean {
                            v - m[d]
                        } else {
                            v
                        }
                    })
                    .collect();

                values.sort_by(|a, b| a.total_cmp(b));

                let zero_fraction = target_sparsity;
                let nonzero_fraction = (1.0 - zero_fraction) / 2.0;

                let low_idx = (nonzero_fraction * num_vectors as f32) as usize;
                let high_idx = ((1.0 - nonzero_fraction) * num_vectors as f32) as usize;

                let low_idx = low_idx.min(num_vectors - 1);
                let high_idx = high_idx.min(num_vectors - 1);

                thresholds.push((values[low_idx], values[high_idx]));
            }

            self.adaptive_thresholds = Some(thresholds);
        }

        Ok(())
    }

    /// Quantize a vector to ternary codes.
    pub fn quantize(&self, vector: &[f32]) -> crate::Result<TernaryVector> {
        if vector.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let centered: Vec<f32> = if let Some(ref mean) = self.mean {
            vector
                .iter()
                .zip(mean.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        } else {
            vector.to_vec()
        };

        let processed: Vec<f32> = if self.config.normalize {
            let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                centered.iter().map(|&x| x / norm).collect()
            } else {
                centered
            }
        } else {
            centered
        };

        let original_norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        let num_bytes = self.dimension.div_ceil(4);
        let mut data = vec![0u8; num_bytes];
        let mut positive_count = 0;
        let mut negative_count = 0;

        for (i, &v) in processed.iter().enumerate() {
            let (thresh_low, thresh_high) = if let Some(ref thresholds) = self.adaptive_thresholds {
                thresholds[i]
            } else {
                (self.config.threshold_low, self.config.threshold_high)
            };

            let bits: u8 = if v > thresh_high {
                positive_count += 1;
                0b01
            } else if v < thresh_low {
                negative_count += 1;
                0b10
            } else {
                0b00
            };

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= bits << bit_offset;
        }

        Ok(TernaryVector {
            data,
            dimension: self.dimension,
            positive_count,
            negative_count,
            original_norm,
        })
    }
}

/// Inner product between two ternary vectors.
///
/// Returns 0 if dimensions mismatch.
#[must_use]
pub fn ternary_inner_product(a: &TernaryVector, b: &TernaryVector) -> i32 {
    if a.dimension != b.dimension {
        return 0;
    }

    let mut sum: i32 = 0;
    for (byte_a, byte_b) in a.data.iter().zip(b.data.iter()) {
        for i in 0..4 {
            let bits_a = (*byte_a >> (i * 2)) & 0b11;
            let bits_b = (*byte_b >> (i * 2)) & 0b11;

            let val_a = match bits_a {
                0b01 => 1i32,
                0b10 => -1,
                _ => 0,
            };
            let val_b = match bits_b {
                0b01 => 1i32,
                0b10 => -1,
                _ => 0,
            };

            sum += val_a * val_b;
        }
    }

    sum
}

/// Cosine similarity between two ternary vectors.
///
/// Returns 0.0 if either vector is all-zero.
#[must_use]
pub fn ternary_cosine_similarity(a: &TernaryVector, b: &TernaryVector) -> f32 {
    let ip = ternary_inner_product(a, b) as f32;

    let norm_a = ((a.positive_count + a.negative_count) as f32).sqrt();
    let norm_b = ((b.positive_count + b.negative_count) as f32).sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (ip / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Asymmetric inner product: f32 query vs ternary codes.
///
/// Returns 0.0 if dimensions mismatch.
#[must_use]
pub fn asymmetric_inner_product(query: &[f32], quantized: &TernaryVector) -> f32 {
    if query.len() != quantized.dimension {
        return 0.0;
    }

    let mut sum = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let val = quantized.get(i);
        sum += q * (val as f32);
    }
    sum
}

/// Asymmetric cosine distance: `1 - cos(query, quantized)`.
///
/// Returns 1.0 if either norm is near zero.
#[must_use]
pub fn asymmetric_cosine_distance(query: &[f32], quantized: &TernaryVector) -> f32 {
    let ip = asymmetric_inner_product(query, quantized);

    let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let ternary_norm = ((quantized.positive_count + quantized.negative_count) as f32).sqrt();

    if query_norm < 1e-10 || ternary_norm < 1e-10 {
        return 1.0;
    }

    1.0 - (ip / (query_norm * ternary_norm))
}

/// Hamming distance between two ternary vectors.
///
/// Counts positions where the ternary values differ.
///
/// Returns `None` if the vectors have different dimensions.
#[must_use]
pub fn ternary_hamming(a: &TernaryVector, b: &TernaryVector) -> Option<usize> {
    if a.dimension != b.dimension {
        return None;
    }

    let mut diff = 0;
    for i in 0..a.dimension {
        if a.get(i) != b.get(i) {
            diff += 1;
        }
    }
    Some(diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_quantization() {
        let quantizer = TernaryQuantizer::with_dimension(8);
        let vector = vec![0.5, -0.5, 0.1, -0.1, 0.8, -0.8, 0.0, 0.2];

        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.dimension(), 8);
        assert!(quantized.memory_bytes() <= 2);
    }

    #[test]
    fn test_ternary_values() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        let vector = vec![0.5, -0.5, 0.1, -0.1];
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.get(0), 1);
        assert_eq!(quantized.get(1), -1);
        assert_eq!(quantized.get(2), 0);
        assert_eq!(quantized.get(3), 0);
    }

    #[test]
    fn test_hamming_distance() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        let v1 = vec![0.5, -0.5, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.0, -0.5];

        let q1 = quantizer.quantize(&v1).unwrap();
        let q2 = quantizer.quantize(&v2).unwrap();

        assert_eq!(ternary_hamming(&q1, &q2), Some(2));
    }

    #[test]
    fn ternary_values_in_range() {
        let dim = 64;
        let tq = TernaryQuantizer::with_dimension(dim);
        let vector = vec![0.5f32; dim];
        let tv = tq.quantize(&vector).unwrap();
        for i in 0..dim {
            let v = tv.get(i);
            assert!(
                v == -1 || v == 0 || v == 1,
                "ternary value {} at index {}",
                v,
                i
            );
        }
    }

    #[test]
    fn ternary_ip_commutative() {
        let dim = 32;
        let config = TernaryConfig {
            normalize: false,
            ..TernaryConfig::default()
        };
        let tq = TernaryQuantizer::new(dim, config);
        let a = tq.quantize(&vec![1.0f32; dim]).unwrap();
        let b = tq.quantize(&vec![-0.5f32; dim]).unwrap();
        let ip_ab = ternary_inner_product(&a, &b);
        let ip_ba = ternary_inner_product(&b, &a);
        assert_eq!(ip_ab, ip_ba, "IP should be commutative");
    }

    #[test]
    fn cosine_similarity_clamped() {
        let config = TernaryConfig {
            normalize: false,
            ..TernaryConfig::default()
        };
        let dim = 8;
        let tq = TernaryQuantizer::new(dim, config);

        // Identical vectors -> similarity = 1.0
        let a = tq.quantize(&vec![1.0; dim]).unwrap();
        let sim = ternary_cosine_similarity(&a, &a);
        assert!(
            (-1.0..=1.0).contains(&sim),
            "identical vectors: sim {sim} out of [-1, 1]"
        );
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should be ~1.0");

        // Opposite vectors -> similarity = -1.0
        let pos = tq.quantize(&vec![1.0; dim]).unwrap();
        let neg = tq.quantize(&vec![-1.0; dim]).unwrap();
        let sim = ternary_cosine_similarity(&pos, &neg);
        assert!(
            (-1.0..=1.0).contains(&sim),
            "opposite vectors: sim {sim} out of [-1, 1]"
        );
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should be ~-1.0");

        // Orthogonal vectors -> similarity = 0.0
        let v1 = tq
            .quantize(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let v2 = tq
            .quantize(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let sim = ternary_cosine_similarity(&v1, &v2);
        assert!(
            (-1.0..=1.0).contains(&sim),
            "orthogonal vectors: sim {sim} out of [-1, 1]"
        );
        assert!(sim.abs() < 1e-6, "orthogonal vectors should be ~0.0");

        // All-zero vector -> returns 0.0 (early exit)
        let zero = tq.quantize(&vec![0.0; dim]).unwrap();
        let sim = ternary_cosine_similarity(&a, &zero);
        assert!(
            (-1.0..=1.0).contains(&sim),
            "zero vector: sim {sim} out of [-1, 1]"
        );
        assert_eq!(sim, 0.0, "similarity with zero vector should be 0.0");
    }

    // ---- error case tests ----

    #[test]
    fn quantize_dimension_mismatch() {
        let tq = TernaryQuantizer::with_dimension(8);
        assert!(tq.quantize(&[1.0f32; 4]).is_err());
    }

    #[test]
    fn fit_dimension_mismatch() {
        let mut tq = TernaryQuantizer::with_dimension(8);
        // 10 floats for 2 vectors of dimension 8 -> mismatch
        assert!(tq.fit(&[1.0f32; 10], 2).is_err());
    }

    #[test]
    fn hamming_dimension_mismatch_returns_none() {
        let config = TernaryConfig {
            normalize: false,
            ..TernaryConfig::default()
        };
        let tq4 = TernaryQuantizer::new(4, config.clone());
        let tq8 = TernaryQuantizer::new(8, config);

        let a = tq4.quantize(&[1.0; 4]).unwrap();
        let b = tq8.quantize(&[1.0; 8]).unwrap();

        assert_eq!(ternary_hamming(&a, &b), None);
    }

    #[test]
    fn inner_product_dimension_mismatch_returns_zero() {
        let config = TernaryConfig {
            normalize: false,
            ..TernaryConfig::default()
        };
        let tq4 = TernaryQuantizer::new(4, config.clone());
        let tq8 = TernaryQuantizer::new(8, config);

        let a = tq4.quantize(&[1.0; 4]).unwrap();
        let b = tq8.quantize(&[1.0; 8]).unwrap();

        assert_eq!(ternary_inner_product(&a, &b), 0);
    }

    #[test]
    fn asymmetric_ip_dimension_mismatch_returns_zero() {
        let tq = TernaryQuantizer::with_dimension(8);
        let q = tq.quantize(&[1.0; 8]).unwrap();
        // query dimension 4 != quantized dimension 8
        assert_eq!(asymmetric_inner_product(&[1.0f32; 4], &q), 0.0);
    }
}
