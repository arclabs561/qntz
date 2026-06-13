//! Rotation-based binary quantization.
//!
//! Applies a random orthogonal rotation to concentrate information, then
//! sign-thresholds each rotated dimension to a single bit. The approach
//! follows the TurboQuant intuition: after rotation, dimensions are
//! approximately independent and equally informative, so 1-bit quantization
//! loses minimal semantic signal.
//!
//! Optionally, `projected_dim < dim` reduces dimensionality at quantization
//! time by taking only the first `projected_dim` rows of the rotation matrix.
//!
//! # Bit convention
//!
//! - bit = 1 if the rotated component is >= 0
//! - bit = 0 otherwise
//!
//! Bits are packed LSB-first into bytes (8 dimensions per byte).
//!
//! # Example
//!
//! ```rust
//! use qntz::binary::BinaryQuantizer;
//!
//! let dim = 32;
//! let q = BinaryQuantizer::new(dim, dim, 42);
//!
//! let vector: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
//! let code = q.quantize(&vector).unwrap();
//!
//! assert_eq!(code.len(), dim.div_ceil(8));
//!
//! // Asymmetric distance: query stays float, database vector is binary.
//! let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
//! let dist = q.asymmetric_distance(&query, &code).unwrap();
//! assert!(dist.is_finite());
//! ```

use crate::VQuantError;

/// Rotation-based binary quantizer.
///
/// Stores a `projected_dim × dim` rotation matrix (row-major). Each row is a
/// unit vector; together the rows form an orthonormal set drawn from a uniform
/// random orthogonal distribution (Gram-Schmidt on a Gaussian matrix).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryQuantizer {
    /// Flattened `projected_dim × dim` rotation matrix (row-major).
    rotation: Vec<f32>,
    /// Input dimensionality.
    dim: usize,
    /// Output dimensionality after rotation (≤ `dim`).
    projected_dim: usize,
}

impl BinaryQuantizer {
    /// Create a new quantizer with a random orthogonal rotation matrix.
    ///
    /// `projected_dim` may be less than `dim` to simultaneously reduce
    /// dimensionality. When `projected_dim == dim` the full square rotation
    /// is used.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid dimensions yield an unusable (zero-dim) state.
    /// Use [`BinaryQuantizer::try_new`] if you need error propagation.
    #[must_use]
    pub fn new(dim: usize, projected_dim: usize, seed: u64) -> Self {
        let projected_dim = projected_dim.min(dim);
        let rotation = generate_rotation(dim, projected_dim, seed);
        Self {
            rotation,
            dim,
            projected_dim,
        }
    }

    /// Fallible constructor — returns an error if `dim` is zero or
    /// `projected_dim` is zero.
    pub fn try_new(dim: usize, projected_dim: usize, seed: u64) -> crate::Result<Self> {
        if dim == 0 {
            return Err(VQuantError::InvalidConfig {
                field: "dim",
                reason: "must be > 0",
            });
        }
        if projected_dim == 0 {
            return Err(VQuantError::InvalidConfig {
                field: "projected_dim",
                reason: "must be > 0",
            });
        }
        Ok(Self::new(dim, projected_dim, seed))
    }

    /// Number of bytes in a packed binary code produced by this quantizer.
    #[must_use]
    pub fn code_len(&self) -> usize {
        self.projected_dim.div_ceil(8)
    }

    /// Input dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Output (projected) dimensionality.
    #[must_use]
    pub fn projected_dim(&self) -> usize {
        self.projected_dim
    }

    /// Quantize a single vector: rotate then sign-threshold to packed bits.
    ///
    /// Returns `projected_dim.div_ceil(8)` bytes, packed LSB-first.
    ///
    /// # Errors
    ///
    /// Returns [`VQuantError::DimensionMismatch`] if `vector.len() != dim`.
    pub fn quantize(&self, vector: &[f32]) -> crate::Result<Vec<u8>> {
        if vector.len() != self.dim {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(rotate_and_pack(
            &self.rotation,
            vector,
            self.dim,
            self.projected_dim,
        ))
    }

    /// Quantize a batch of vectors.
    ///
    /// Each slice in `vectors` must have length `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`VQuantError::DimensionMismatch`] on the first vector whose
    /// length does not match `dim`.
    pub fn quantize_batch(&self, vectors: &[&[f32]]) -> crate::Result<Vec<Vec<u8>>> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }

    /// Asymmetric distance between a raw query and a binary-coded database vector.
    ///
    /// Rotates the query to float coordinates, then computes the inner product
    /// against the {-1, +1} interpretation of the binary code. Returns
    /// `-(sum of matching signs)` normalised to `[-1, 1]` — lower is closer
    /// (larger inner product means more similar). Specifically, the return value
    /// is:
    ///
    /// ```text
    /// distance = -( Σ_i  rotated_query[i] * sign(code_bit[i]) ) / projected_dim
    /// ```
    ///
    /// where sign maps bit=1 to +1 and bit=0 to -1.
    ///
    /// # Errors
    ///
    /// Returns [`VQuantError::DimensionMismatch`] if `query.len() != dim` or
    /// `code.len() < code_len()`.
    #[allow(clippy::needless_range_loop)]
    pub fn asymmetric_distance(&self, query: &[f32], code: &[u8]) -> crate::Result<f32> {
        if query.len() != self.dim {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let required = self.code_len();
        if code.len() < required {
            return Err(VQuantError::DimensionMismatch {
                expected: required,
                got: code.len(),
            });
        }

        // Rotate query to projected space.
        let rotated = apply_rotation_rect(&self.rotation, query, self.dim, self.projected_dim);

        // Inner product against +/-1 codes.
        // Process 8 elements per byte to avoid per-element div/mod.
        let mut ip = 0.0f32;
        let full_bytes = rotated.len() / 8;
        for byte_idx in 0..full_bytes {
            let byte = code[byte_idx];
            let base = byte_idx * 8;
            // Unroll 8 bits per byte: sign = +1 if bit set, -1 otherwise
            ip += if byte & 1 != 0 {
                rotated[base]
            } else {
                -rotated[base]
            };
            ip += if byte & 2 != 0 {
                rotated[base + 1]
            } else {
                -rotated[base + 1]
            };
            ip += if byte & 4 != 0 {
                rotated[base + 2]
            } else {
                -rotated[base + 2]
            };
            ip += if byte & 8 != 0 {
                rotated[base + 3]
            } else {
                -rotated[base + 3]
            };
            ip += if byte & 16 != 0 {
                rotated[base + 4]
            } else {
                -rotated[base + 4]
            };
            ip += if byte & 32 != 0 {
                rotated[base + 5]
            } else {
                -rotated[base + 5]
            };
            ip += if byte & 64 != 0 {
                rotated[base + 6]
            } else {
                -rotated[base + 6]
            };
            ip += if byte & 128 != 0 {
                rotated[base + 7]
            } else {
                -rotated[base + 7]
            };
        }
        // Tail elements
        for i in (full_bytes * 8)..rotated.len() {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (code[byte_idx] >> bit_idx) & 1;
            ip += if bit == 1 { rotated[i] } else { -rotated[i] };
        }

        // Negate so that similar vectors have smaller distance.
        Ok(-ip / self.projected_dim as f32)
    }

    /// Symmetric Hamming distance between two packed binary codes.
    ///
    /// Counts the number of differing bits across the full byte slices.
    /// Both slices must have the same length (typically [`code_len()`](Self::code_len)).
    #[must_use]
    pub fn symmetric_distance(code_a: &[u8], code_b: &[u8]) -> u32 {
        crate::simd_ops::hamming_distance(code_a, code_b)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Generate a `projected_dim × dim` row-major orthonormal matrix via
/// Gram-Schmidt on Gaussian random vectors.
fn generate_rotation(dim: usize, projected_dim: usize, seed: u64) -> Vec<f32> {
    crate::rotation::orthogonal_rotation_matrix(dim, projected_dim, seed)
}

/// Matrix-vector product: `projected_dim × dim` matrix times `dim`-vector.
fn apply_rotation_rect(
    rotation: &[f32],
    vector: &[f32],
    dim: usize,
    projected_dim: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; projected_dim];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dim..(i + 1) * dim];
        #[cfg(feature = "simd")]
        {
            *out = innr::dot(row, vector);
        }
        #[cfg(not(feature = "simd"))]
        {
            let chunks = dim / 4;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            for c in 0..chunks {
                let b = c * 4;
                s0 += row[b] * vector[b];
                s1 += row[b + 1] * vector[b + 1];
                s2 += row[b + 2] * vector[b + 2];
                s3 += row[b + 3] * vector[b + 3];
            }
            let mut sum = (s0 + s1) + (s2 + s3);
            for j in (chunks * 4)..dim {
                sum += row[j] * vector[j];
            }
            *out = sum;
        }
    }
    result
}

/// Rotate a vector and pack the sign bits into bytes (LSB-first).
fn rotate_and_pack(rotation: &[f32], vector: &[f32], dim: usize, projected_dim: usize) -> Vec<u8> {
    let bytes_needed = projected_dim.div_ceil(8);
    let mut packed = vec![0u8; bytes_needed];

    for i in 0..projected_dim {
        let row_start = i * dim;
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[row_start + j] * vector[j];
        }
        if sum >= 0.0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }

    packed
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- construction ----

    #[test]
    fn try_new_zero_dim_rejected() {
        assert!(BinaryQuantizer::try_new(0, 4, 42).is_err());
    }

    #[test]
    fn try_new_zero_projected_rejected() {
        assert!(BinaryQuantizer::try_new(8, 0, 42).is_err());
    }

    #[test]
    fn projected_dim_clamped_to_dim() {
        let q = BinaryQuantizer::new(8, 100, 42);
        assert_eq!(q.projected_dim(), 8);
    }

    // ---- code length ----

    #[test]
    fn code_len_exact_multiple() {
        let q = BinaryQuantizer::new(16, 16, 0);
        assert_eq!(q.code_len(), 2);
    }

    #[test]
    fn code_len_non_multiple() {
        let q = BinaryQuantizer::new(16, 9, 0);
        assert_eq!(q.code_len(), 2); // 9 bits -> 2 bytes
    }

    // ---- quantize ----

    #[test]
    fn quantize_output_length() {
        let dim = 32;
        let q = BinaryQuantizer::new(dim, dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let code = q.quantize(&v).unwrap();
        assert_eq!(code.len(), dim.div_ceil(8));
    }

    #[test]
    fn quantize_projected_output_length() {
        let dim = 32;
        let proj = 12;
        let q = BinaryQuantizer::new(dim, proj, 7);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let code = q.quantize(&v).unwrap();
        assert_eq!(code.len(), proj.div_ceil(8));
    }

    #[test]
    fn quantize_dimension_mismatch() {
        let q = BinaryQuantizer::new(16, 16, 0);
        assert!(q.quantize(&[1.0f32; 8]).is_err());
    }

    #[test]
    fn quantize_batch_matches_individual() {
        let dim = 16;
        let q = BinaryQuantizer::new(dim, dim, 99);
        let v1: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..dim).map(|i| -(i as f32)).collect();

        let batch = q.quantize_batch(&[&v1, &v2]).unwrap();
        let single1 = q.quantize(&v1).unwrap();
        let single2 = q.quantize(&v2).unwrap();

        assert_eq!(batch[0], single1);
        assert_eq!(batch[1], single2);
    }

    // ---- symmetric_distance ----

    #[test]
    fn symmetric_distance_identical_codes_zero() {
        let code = vec![0b10101010u8, 0b11001100];
        assert_eq!(BinaryQuantizer::symmetric_distance(&code, &code), 0);
    }

    #[test]
    fn symmetric_distance_all_flipped() {
        let a = vec![0u8; 2];
        let b = vec![0xFFu8; 2];
        assert_eq!(BinaryQuantizer::symmetric_distance(&a, &b), 16);
    }

    #[test]
    fn symmetric_matches_manual_popcount() {
        let a = vec![0b00001111u8];
        let b = vec![0b11110000u8];
        // XOR = 0xFF -> 8 bits set
        assert_eq!(BinaryQuantizer::symmetric_distance(&a, &b), 8);
    }

    // ---- asymmetric_distance ----

    #[test]
    fn asymmetric_distance_dimension_mismatch() {
        let q = BinaryQuantizer::new(16, 16, 0);
        let code = vec![0u8; q.code_len()];
        assert!(q.asymmetric_distance(&[1.0f32; 8], &code).is_err());
    }

    #[test]
    fn asymmetric_distance_code_too_short() {
        let q = BinaryQuantizer::new(16, 16, 0);
        let query = vec![0.0f32; 16];
        assert!(q.asymmetric_distance(&query, &[0u8; 1]).is_err());
    }

    #[test]
    fn asymmetric_distance_finite() {
        let dim = 32;
        let q = BinaryQuantizer::new(dim, dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let code = q.quantize(&v).unwrap();
        let dist = q.asymmetric_distance(&v, &code).unwrap();
        assert!(dist.is_finite());
    }

    // ---- relative ordering (key correctness property) ----
    //
    // A vector close to the query should have a smaller asymmetric distance
    // than a vector orthogonal or opposite to the query.
    #[test]
    fn asymmetric_distance_preserves_relative_ordering() {
        let dim = 64;
        let q = BinaryQuantizer::new(dim, dim, 1337);

        // query: all ones (normalised)
        let query: Vec<f32> = vec![1.0f32 / (dim as f32).sqrt(); dim];

        // close: same direction, slight noise
        let close: Vec<f32> = (0..dim)
            .map(|i| 1.0f32 / (dim as f32).sqrt() + (i as f32) * 1e-4)
            .collect();

        // far: opposite direction
        let far: Vec<f32> = vec![-1.0f32 / (dim as f32).sqrt(); dim];

        let code_close = q.quantize(&close).unwrap();
        let code_far = q.quantize(&far).unwrap();

        let dist_close = q.asymmetric_distance(&query, &code_close).unwrap();
        let dist_far = q.asymmetric_distance(&query, &code_far).unwrap();

        assert!(
            dist_close < dist_far,
            "close distance {dist_close} should be less than far distance {dist_far}"
        );
    }

    // ---- symmetric Hamming agrees with asymmetric ordering ----
    //
    // If we encode both query and a candidate, the Hamming distance between
    // their codes should mirror the asymmetric distance ordering.
    #[test]
    fn symmetric_distance_ordering_consistent_with_asymmetric() {
        let dim = 64;
        let q = BinaryQuantizer::new(dim, dim, 2024);

        let query: Vec<f32> = vec![1.0f32 / (dim as f32).sqrt(); dim];
        let close: Vec<f32> = vec![0.9f32 / (dim as f32).sqrt(); dim];
        let far: Vec<f32> = vec![-1.0f32 / (dim as f32).sqrt(); dim];

        let code_q = q.quantize(&query).unwrap();
        let code_close = q.quantize(&close).unwrap();
        let code_far = q.quantize(&far).unwrap();

        let sym_close = BinaryQuantizer::symmetric_distance(&code_q, &code_close);
        let sym_far = BinaryQuantizer::symmetric_distance(&code_q, &code_far);

        assert!(
            sym_close < sym_far,
            "symmetric close {sym_close} should be less than far {sym_far}"
        );
    }

    // ---- rotation orthogonality sanity ----
    //
    // Each row of the rotation matrix should be a unit vector, and distinct
    // rows should be nearly orthogonal.
    #[test]
    fn rotation_rows_are_unit_vectors() {
        let dim = 8;
        let q = BinaryQuantizer::new(dim, dim, 0);
        for row in 0..dim {
            let start = row * dim;
            let norm_sq: f32 = q.rotation[start..start + dim].iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-5,
                "row {row} norm^2 = {norm_sq}, expected ~1.0"
            );
        }
    }

    #[test]
    fn rotation_rows_are_orthogonal() {
        let dim = 8;
        let q = BinaryQuantizer::new(dim, dim, 0);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let ri = &q.rotation[i * dim..(i + 1) * dim];
                let rj = &q.rotation[j * dim..(j + 1) * dim];
                let dot: f32 = ri.iter().zip(rj.iter()).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-5,
                    "rows {i} and {j} dot product = {dot}, expected ~0"
                );
            }
        }
    }
}
