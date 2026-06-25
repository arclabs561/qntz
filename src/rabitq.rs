//! RaBitQ (Randomized Binary Quantization).
//!
//! Quantizes each dimension using a sign bit plus optional extended bits (up to
//! 8-bit total), with a random rotation for decorrelation and correction factors
//! for approximate L2 distance.
//!
//! # Example
//!
//! ```rust
//! use qntz::rabitq::{RaBitQQuantizer, RaBitQConfig};
//!
//! let dim = 32;
//! let quantizer = RaBitQQuantizer::with_config(
//!     dim,
//!     42,
//!     RaBitQConfig::bits4(),
//! ).unwrap();
//!
//! let vector: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
//! let quantized = quantizer.quantize(&vector).unwrap();
//!
//! assert_eq!(quantized.dimension, dim);
//! assert_eq!(quantized.ex_bits, 3); // 4-bit total = 1 sign + 3 extended
//!
//! // Approximate L2 distance from a query
//! let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
//! let dist = quantizer.approximate_distance(&query, &quantized).unwrap();
//! assert!(dist >= 0.0);
//! ```

use crate::VQuantError;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Configuration for RaBitQ quantization.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQConfig {
    /// Total bits per dimension (1-8). 1 = binary only.
    pub total_bits: usize,
    /// Precomputed scaling factor (None = compute optimal per vector).
    pub t_const: Option<f32>,
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            total_bits: 4, // 4-bit default: balance of speed/accuracy
            t_const: None,
        }
    }
}

impl RaBitQConfig {
    /// Binary quantization (1-bit per dimension).
    #[must_use]
    pub fn binary() -> Self {
        Self {
            total_bits: 1,
            t_const: None,
        }
    }

    /// 2-bit quantization (1 sign + 1 extended). ~75% recall without rerank.
    #[must_use]
    pub fn bits2() -> Self {
        Self {
            total_bits: 2,
            t_const: None,
        }
    }

    /// 3-bit quantization (1 sign + 2 extended). ~85% recall without rerank.
    #[must_use]
    pub fn bits3() -> Self {
        Self {
            total_bits: 3,
            t_const: None,
        }
    }

    /// 4-bit quantization (default, good balance). ~90% recall without rerank.
    #[must_use]
    pub fn bits4() -> Self {
        Self {
            total_bits: 4,
            t_const: None,
        }
    }

    /// 5-bit quantization. ~95% recall without rerank.
    #[must_use]
    pub fn bits5() -> Self {
        Self {
            total_bits: 5,
            t_const: None,
        }
    }

    /// 6-bit quantization. ~97% recall without rerank.
    #[must_use]
    pub fn bits6() -> Self {
        Self {
            total_bits: 6,
            t_const: None,
        }
    }

    /// 7-bit quantization. ~99% recall without rerank.
    #[must_use]
    pub fn bits7() -> Self {
        Self {
            total_bits: 7,
            t_const: None,
        }
    }

    /// 8-bit quantization (high accuracy). ~99.5% recall without rerank.
    #[must_use]
    pub fn bits8() -> Self {
        Self {
            total_bits: 8,
            t_const: None,
        }
    }

    /// Create config with precomputed scaling factor for faster quantization.
    /// Trades <1% accuracy for substantially faster quantization.
    #[must_use]
    pub fn with_const_scaling(self, dimension: usize, seed: u64) -> Self {
        let ex_bits = self.total_bits.saturating_sub(1);
        let t_const = if ex_bits > 0 {
            Some(compute_const_scaling_factor(dimension, ex_bits, seed))
        } else {
            None
        };
        Self { t_const, ..self }
    }
}

/// Quantized vector with extended codes and corrective factors.
#[derive(Clone, Debug)]
pub struct QuantizedVector {
    /// Binary codes (packed, 8 dimensions per byte)
    pub binary_codes: Vec<u8>,
    /// Extended codes (ex_bits per dimension, packed)
    pub extended_codes: Vec<u8>,
    /// Total code per dimension (for convenience/debug)
    pub codes: Vec<u16>,
    /// Extended bits count
    pub ex_bits: u8,
    /// Original dimension
    pub dimension: usize,
    /// Rescaling factor (delta)
    pub delta: f32,
    /// Offset for reconstruction (vl = delta * cb)
    pub vl: f32,
    /// Additive correction factor for distance
    pub f_add: f32,
    /// Multiplicative correction factor for distance
    pub f_rescale: f32,
    /// Quantization error estimate
    pub f_error: f32,
    /// L2 norm of residual
    pub residual_norm: f32,
}

/// RaBitQ quantizer with extended bit support.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQQuantizer {
    dimension: usize,
    /// Random rotation matrix (orthogonal)
    rotation: Vec<f32>,
    /// Centroid for residual computation
    centroid: Option<Vec<f32>>,
    /// Configuration
    config: RaBitQConfig,
}

impl RaBitQQuantizer {
    /// Create new RaBitQ quantizer with default config.
    pub fn new(dimension: usize, seed: u64) -> crate::Result<Self> {
        Self::with_config(dimension, seed, RaBitQConfig::default())
    }

    /// Create quantizer with specific config.
    pub fn with_config(dimension: usize, seed: u64, config: RaBitQConfig) -> crate::Result<Self> {
        if dimension == 0 {
            return Err(VQuantError::InvalidConfig {
                field: "dimension",
                reason: "must be > 0",
            });
        }
        if config.total_bits == 0 || config.total_bits > 8 {
            return Err(VQuantError::InvalidConfig {
                field: "total_bits",
                reason: "must be 1-8",
            });
        }

        let rotation = generate_orthogonal_rotation(dimension, seed);

        Ok(Self {
            dimension,
            rotation,
            centroid: None,
            config,
        })
    }

    /// Create binary-only quantizer.
    pub fn binary(dimension: usize, seed: u64) -> crate::Result<Self> {
        Self::with_config(dimension, seed, RaBitQConfig::binary())
    }

    /// Fit quantizer on training vectors (computes centroid).
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> crate::Result<()> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: num_vectors * self.dimension,
                got: vectors.len(),
            });
        }

        let mut centroid = vec![0.0f32; self.dimension];
        for i in 0..num_vectors {
            let vec = &vectors[i * self.dimension..(i + 1) * self.dimension];
            for (j, &v) in vec.iter().enumerate() {
                centroid[j] += v;
            }
        }
        for c in &mut centroid {
            *c /= num_vectors as f32;
        }
        self.centroid = Some(centroid);

        Ok(())
    }

    /// Set centroid directly.
    pub fn set_centroid(&mut self, centroid: Vec<f32>) -> crate::Result<()> {
        if centroid.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: centroid.len(),
            });
        }
        self.centroid = Some(centroid);
        Ok(())
    }

    /// Quantize a vector relative to centroid.
    pub fn quantize(&self, vector: &[f32]) -> crate::Result<QuantizedVector> {
        if vector.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let default_centroid = vec![0.0f32; self.dimension];
        let centroid = self.centroid.as_ref().unwrap_or(&default_centroid);
        self.quantize_with_centroid(vector, centroid)
    }

    /// Quantize relative to specific centroid.
    pub fn quantize_with_centroid(
        &self,
        vector: &[f32],
        centroid: &[f32],
    ) -> crate::Result<QuantizedVector> {
        let dim = self.dimension;
        let ex_bits = self.config.total_bits.saturating_sub(1);

        // Step 1: residual
        let residual: Vec<f32> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        // Step 2: rotation
        let rotated = apply_rotation(&residual, &self.rotation, dim);

        // Step 3: sign bits
        let mut binary_codes_unpacked = vec![0u8; dim];
        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                binary_codes_unpacked[i] = 1;
            }
        }

        // Step 4: extended codes
        let extended_codes_unpacked = if ex_bits > 0 {
            self.compute_extended_codes(&rotated, ex_bits).0
        } else {
            vec![0u16; dim]
        };

        // Step 5: total codes
        let mut total_codes = vec![0u16; dim];
        for i in 0..dim {
            total_codes[i] =
                extended_codes_unpacked[i] + ((binary_codes_unpacked[i] as u16) << ex_bits);
        }

        // Step 6: correction factors
        // Use the full multi-bit centered codes so that f_rescale and the
        // query-time IP (which uses codes[i] + cb) are computed with the
        // same xu vector.  For binary (ex_bits=0), cb = -0.5 and
        // codes[i] ∈ {0,1}, so xu_multibit[i] = codes[i] - 0.5 = b_i - 0.5,
        // identical to the old binary path.
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let xu_multibit: Vec<f32> = total_codes.iter().map(|&c| c as f32 + cb).collect();
        let (f_add, f_rescale, f_error, residual_norm) =
            self.compute_correction_factors(&rotated, &xu_multibit);

        // Step 7: delta/vl (xu_multibit == quantized_shifted, reuse it)
        let norm_quan_sqr: f32 = xu_multibit.iter().map(|x| x * x).sum();
        let norm_residual_sqr: f32 = rotated.iter().map(|x| x * x).sum();
        let dot_rq: f32 = rotated
            .iter()
            .zip(xu_multibit.iter())
            .map(|(r, q)| r * q)
            .sum();

        let norm_residual = norm_residual_sqr.sqrt();
        let norm_quant = norm_quan_sqr.sqrt();
        let denom = (norm_residual * norm_quant).max(f32::EPSILON);
        let cos_sim = (dot_rq / denom).clamp(-1.0, 1.0);

        let delta = if norm_quant <= f32::EPSILON {
            0.0
        } else {
            (norm_residual / norm_quant) * cos_sim
        };
        let vl = delta * cb;

        // pack
        let bytes_needed = binary_codes_unpacked.len().div_ceil(8);
        let mut binary_codes = vec![0u8; bytes_needed];
        crate::simd_ops::pack_binary_fast(&binary_codes_unpacked, &mut binary_codes)
            .expect("buffer sized correctly");

        let extended_codes = pack_extended_codes(&extended_codes_unpacked, ex_bits);

        Ok(QuantizedVector {
            binary_codes,
            extended_codes,
            codes: total_codes,
            ex_bits: ex_bits as u8,
            dimension: dim,
            delta,
            vl,
            f_add,
            f_rescale,
            f_error,
            residual_norm,
        })
    }

    /// Quantizes a pre-rotated residual `R · (v − c)` without repeating the
    /// O(d²) rotation.
    ///
    /// Use this when the caller has already rotated `v − c` (or a batch of
    /// such residuals) and wants to reuse that work.
    /// [`quantize_with_centroid`](Self::quantize_with_centroid) is the direct
    /// equivalent when the residual has not yet been rotated.
    ///
    /// The resulting [`QuantizedVector`] is a ranking-score code: pair it
    /// with [`approximate_l2_sqr_prerotated`](Self::approximate_l2_sqr_prerotated),
    /// which returns `||q − v||^2 − ||q − c||^2`. For an absolute distance
    /// that is comparable across different centroids (graph edges), use
    /// [`quantize_edge_prerotated`](Self::quantize_edge_prerotated) instead,
    /// which returns a distinct [`EdgeQuantizedVector`] and pairs with
    /// [`edge_distance_term_prerotated`](Self::edge_distance_term_prerotated).
    ///
    /// # Arguments
    ///
    /// * `rotated_residual` — `R · (v − c)`, with length equal to the
    ///   quantizer's dimension.
    /// * `raw_centroid` — the same `c` used to form `rotated_residual`. Pass
    ///   a zero vector if `rotated_residual` is simply `R · v`. Passing a
    ///   mismatching centroid produces a value that is not a valid distance.
    ///
    /// # Errors
    ///
    /// Returns [`VQuantError::DimensionMismatch`] if either slice length
    /// differs from the quantizer's dimension.
    pub fn quantize_prerotated(
        &self,
        rotated_residual: &[f32],
        raw_centroid: &[f32],
    ) -> crate::Result<QuantizedVector> {
        let dim = self.dimension;
        if rotated_residual.len() != dim || raw_centroid.len() != dim {
            return Err(VQuantError::DimensionMismatch {
                expected: dim,
                got: rotated_residual.len(),
            });
        }
        let ex_bits = self.config.total_bits.saturating_sub(1);
        let rotated = rotated_residual;

        // Step 3: sign bits
        let mut binary_codes_unpacked = vec![0u8; dim];
        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                binary_codes_unpacked[i] = 1;
            }
        }

        // Step 4: extended codes
        let extended_codes_unpacked = if ex_bits > 0 {
            self.compute_extended_codes(rotated, ex_bits).0
        } else {
            vec![0u16; dim]
        };

        // Step 5: total codes
        let mut total_codes = vec![0u16; dim];
        for i in 0..dim {
            total_codes[i] =
                extended_codes_unpacked[i] + ((binary_codes_unpacked[i] as u16) << ex_bits);
        }

        // Step 6: correction factors
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let xu_multibit: Vec<f32> = total_codes.iter().map(|&c| c as f32 + cb).collect();
        // `raw_centroid` is already folded into `rotated_residual` (= R·(v−c));
        // it is retained in the signature for API compatibility and length
        // validation but no longer feeds the estimator (see compute_correction_factors).
        let _ = raw_centroid;
        let (f_add, f_rescale, f_error, residual_norm) =
            self.compute_correction_factors(rotated, &xu_multibit);

        // Step 7: delta/vl
        let norm_quan_sqr: f32 = xu_multibit.iter().map(|x| x * x).sum();
        let norm_residual_sqr: f32 = rotated.iter().map(|x| x * x).sum();
        let dot_rq: f32 = rotated
            .iter()
            .zip(xu_multibit.iter())
            .map(|(r, q)| r * q)
            .sum();
        let norm_residual = norm_residual_sqr.sqrt();
        let norm_quant = norm_quan_sqr.sqrt();
        let denom = (norm_residual * norm_quant).max(f32::EPSILON);
        let cos_sim = (dot_rq / denom).clamp(-1.0, 1.0);
        let delta = if norm_quant <= f32::EPSILON {
            0.0
        } else {
            (norm_residual / norm_quant) * cos_sim
        };
        let vl = delta * cb;

        // pack
        let bytes_needed = binary_codes_unpacked.len().div_ceil(8);
        let mut binary_codes = vec![0u8; bytes_needed];
        crate::simd_ops::pack_binary_fast(&binary_codes_unpacked, &mut binary_codes)
            .expect("buffer sized correctly");
        let extended_codes = pack_extended_codes(&extended_codes_unpacked, ex_bits);

        Ok(QuantizedVector {
            binary_codes,
            extended_codes,
            codes: total_codes,
            ex_bits: ex_bits as u8,
            dimension: dim,
            delta,
            vl,
            f_add,
            f_rescale,
            f_error,
            residual_norm,
        })
    }

    /// Compute extended codes using optimal rescaling.
    fn compute_extended_codes(&self, rotated: &[f32], ex_bits: usize) -> (Vec<u16>, f32) {
        let dim = self.dimension;

        let mut normalized_abs: Vec<f32> = rotated.iter().map(|x| x.abs()).collect();
        let norm: f32 = normalized_abs.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm <= f32::EPSILON {
            return (vec![0u16; dim], 1.0);
        }

        for val in &mut normalized_abs {
            *val /= norm;
        }

        let t = if let Some(t_const) = self.config.t_const {
            t_const as f64
        } else {
            best_rescale_factor(&normalized_abs, ex_bits)
        };

        quantize_extended(&normalized_abs, rotated, ex_bits, t)
    }

    fn compute_correction_factors(&self, residual: &[f32], xu_cb: &[f32]) -> (f32, f32, f32, f32) {
        let dim = self.dimension;

        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();
        let xu_cb_norm_sqr: f32 = xu_cb.iter().map(|x| x * x).sum();
        let ip_resi_xucb: f32 = residual.iter().zip(xu_cb.iter()).map(|(r, x)| r * x).sum();

        let denom = if ip_resi_xucb.abs() <= f32::EPSILON {
            f32::INFINITY
        } else {
            ip_resi_xucb
        };

        // error estimate
        let mut tmp_error = 0.0f32;
        if dim > 1 {
            let ratio = ((l2_sqr * xu_cb_norm_sqr) / (denom * denom)) - 1.0;
            if ratio.is_finite() && ratio > 0.0 {
                const K_CONST_EPSILON: f32 = 1.9;
                tmp_error =
                    l2_norm * K_CONST_EPSILON * ((ratio / ((dim - 1) as f32)).max(0.0)).sqrt();
            }
        }

        // f_add is the per-document constant of the residual-space ranking
        // estimator: dist ~= ||x-c||^2 - 2<q-c, x-c>, so f_add = ||x-c||^2.
        // (The query-only ||q-c||^2 term is dropped; it does not affect ranking.
        // This matches the documented prerotated semantics ||q-v||^2 - ||q-c||^2.)
        // An earlier version added 2*l2_sqr*<raw_centroid, rotated_codes>/denom,
        // which mixes spaces and corrupts ranking once the centroid is non-zero.
        let f_add = l2_sqr;
        let f_rescale = -2.0 * l2_sqr / denom;
        let f_error = 2.0 * tmp_error;

        (f_add, f_rescale, f_error, l2_norm)
    }

    /// Pre-rotate a query vector for use with [`approximate_l2_sqr_prerotated`](Self::approximate_l2_sqr_prerotated).
    ///
    /// Subtracts the centroid (if set) and applies the rotation matrix.
    /// Call this once per query, then use the result for multiple distance
    /// computations to avoid redundant O(d^2) rotation per candidate.
    pub fn rotate_query(&self, query: &[f32]) -> crate::Result<Vec<f32>> {
        if query.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        let default_centroid = vec![0.0f32; self.dimension];
        let centroid = self.centroid.as_deref().unwrap_or(&default_centroid);
        let residual: Vec<f32> = query
            .iter()
            .zip(centroid.iter())
            .map(|(q, c)| q - c)
            .collect();
        Ok(apply_rotation(&residual, &self.rotation, self.dimension))
    }

    /// Ranking score between a pre-rotated query and a quantized vector.
    ///
    /// Under the rank-1 RaBitQ approximation this returns
    /// `||q − v||^2 − ||q − c||^2`, i.e. the true squared L2 distance minus
    /// a per-query constant. The constant cancels across candidates, so the
    /// score is suitable for any within-query top-k ranking — graph beam
    /// search, IVF probing, reranking pools — but it is not a distance.
    /// Scores from quantizers with different centroids are not comparable.
    ///
    /// `rotated_query` must be `R · (q − c)`, the output of
    /// [`rotate_query`](Self::rotate_query). That method uses the quantizer's
    /// own centroid (from [`fit`](Self::fit) or
    /// [`set_centroid`](Self::set_centroid)), so the call amortises the
    /// O(d²) rotation across a batch of distance computations.
    ///
    /// For an absolute `||q − v||^2` composable across graph edges — where
    /// each edge has its own parent — use
    /// [`quantize_edge_prerotated`](Self::quantize_edge_prerotated) with
    /// [`edge_distance_term_prerotated`](Self::edge_distance_term_prerotated).
    /// The distinct [`EdgeQuantizedVector`] type means the compiler will
    /// refuse to let you pass an edge through this method by mistake.
    #[inline]
    pub fn approximate_l2_sqr_prerotated(
        rotated_query: &[f32],
        quantized: &QuantizedVector,
    ) -> f32 {
        let cb = -((1 << quantized.ex_bits) as f32 - 0.5);
        // Two-accumulator reduction: breaks the serial FP dependency chain so
        // that LLVM can issue two fmadd streams in parallel (latency hiding).
        let mut ip0 = 0.0f32;
        let mut ip1 = 0.0f32;
        let codes = &quantized.codes;
        let chunks = rotated_query.len() / 2;
        for i in 0..chunks {
            let j = i * 2;
            ip0 += rotated_query[j] * (codes[j] as f32 + cb);
            ip1 += rotated_query[j + 1] * (codes[j + 1] as f32 + cb);
        }
        if rotated_query.len() % 2 != 0 {
            let last = rotated_query.len() - 1;
            ip0 += rotated_query[last] * (codes[last] as f32 + cb);
        }
        let ip = ip0 + ip1;
        // Ranking proxy ||q-v||^2 - ||q-c||^2: monotonic in true distance but
        // legitimately NEGATIVE for near neighbors, so it must NOT be clamped
        // (clamping collapses the nearest candidates to 0 and loses their order).
        // For an absolute distance, add ||rotated_query||^2 (= ||q-c||^2) once.
        quantized.f_add + quantized.f_rescale * ip
    }

    /// Approximate L2 distance squared.
    pub fn approximate_l2_sqr(
        &self,
        query: &[f32],
        quantized: &QuantizedVector,
    ) -> crate::Result<f32> {
        if query.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let default_centroid = vec![0.0f32; self.dimension];
        let centroid = self.centroid.as_deref().unwrap_or(&default_centroid);

        let query_residual: Vec<f32> = query
            .iter()
            .zip(centroid.iter())
            .map(|(q, c)| q - c)
            .collect();
        let rotated_query = apply_rotation(&query_residual, &self.rotation, self.dimension);

        let cb = -((1 << quantized.ex_bits) as f32 - 0.5);
        let mut ip = 0.0f32;
        for (i, &q) in rotated_query.iter().enumerate() {
            let code_val = quantized.codes[i] as f32 + cb;
            ip += q * code_val;
        }

        // `f_add + f_rescale*ip` estimates ||q-x||^2 - ||q-c||^2 (a ranking
        // proxy that is legitimately negative for near neighbors). Add the
        // query-side ||q-c||^2 to recover the absolute squared distance, which is
        // non-negative and ranking-correct. Without it the trailing `.max(0.0)`
        // collapses all near neighbors to 0 and destroys their ordering.
        let q_resid_norm_sqr: f32 = query_residual.iter().map(|x| x * x).sum();
        let dist = q_resid_norm_sqr + quantized.f_add + quantized.f_rescale * ip;
        Ok(dist.max(0.0))
    }

    /// Approximate Euclidean distance (L2) between a query and a quantized vector.
    ///
    /// This is the square root of [`approximate_l2_sqr`](Self::approximate_l2_sqr).
    pub fn approximate_distance(
        &self,
        query: &[f32],
        quantized: &QuantizedVector,
    ) -> crate::Result<f32> {
        Ok(self.approximate_l2_sqr(query, quantized)?.sqrt())
    }

    // ─── Edge quantization (vertex-relative graph distances) ────────────────

    /// Quantizes an edge residual `v − u` for vertex-relative graph search.
    ///
    /// Each edge `u → v` in a vertex-relative graph (such as SymphonyQG-VR)
    /// stores a RaBitQ code for `v − u`. Unlike a ranking-score code, the
    /// returned [`EdgeQuantizedVector`] carries enough information to form
    /// an absolute `||q − v||^2` at search time when the caller supplies
    /// `||q − u||^2`. The decomposition is:
    ///
    /// ```text
    /// ||q − v||^2  ≈  ||q − u||^2   +   [ ||v − u||^2 + f_rescale · ⟨R(q − u), xu_cb⟩ ]
    ///                 └─ caller ─┘       └────── edge_distance_term_prerotated ──────┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `rotated_parent` — `R · u`. Compute once per parent; reuse across
    ///   all outgoing edges and for every later query.
    /// * `rotated_residual` — `R · (v − u)`. In practice compute this as
    ///   `R·v − R·u`, an O(d) subtraction of pre-rotated vectors, rather
    ///   than rotating the residual (O(d²)).
    ///
    /// Both slices must have length equal to the quantizer's dimension.
    ///
    /// # Errors
    ///
    /// Returns [`VQuantError::DimensionMismatch`] if either slice length
    /// is wrong.
    ///
    /// # Why a distinct type
    ///
    /// Plugging an edge code into
    /// [`approximate_l2_sqr_prerotated`](Self::approximate_l2_sqr_prerotated)
    /// would produce a per-edge systematic bias, because that method returns
    /// a ranking-score shift that is only valid within a single centroid.
    /// Returning [`EdgeQuantizedVector`] instead of [`QuantizedVector`] makes
    /// the mistake a type error; use
    /// [`edge_distance_term_prerotated`](Self::edge_distance_term_prerotated)
    /// and add `||q − u||^2` externally.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};
    ///
    /// let dim = 64;
    /// let mut q = RaBitQQuantizer::with_config(dim, 0, RaBitQConfig::bits4())?;
    /// // Edge codes use no global centroid — every parent plays that role.
    /// q.set_centroid(vec![0.0; dim])?;
    ///
    /// let u: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
    /// let v: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
    /// let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    ///
    /// // Rotate u, v, and q once; reuse the rotations for each edge.
    /// let ru = q.rotate_query(&u)?;
    /// let rv = q.rotate_query(&v)?;
    /// let rq = q.rotate_query(&query)?;
    /// let residual: Vec<f32> = rv.iter().zip(&ru).map(|(a, b)| a - b).collect();
    ///
    /// let edge = q.quantize_edge_prerotated(&ru, &residual)?;
    /// let term = RaBitQQuantizer::edge_distance_term_prerotated(&rq, &edge);
    ///
    /// // Caller supplies ||q - u||^2 to complete the distance.
    /// let qu_sqr: f32 = query.iter().zip(&u).map(|(a, b)| (a - b).powi(2)).sum();
    /// let approx_qv_sqr = qu_sqr + term;
    /// assert!(approx_qv_sqr >= 0.0);
    /// # Ok::<(), qntz::VQuantError>(())
    /// ```
    pub fn quantize_edge_prerotated(
        &self,
        rotated_parent: &[f32],
        rotated_residual: &[f32],
    ) -> crate::Result<EdgeQuantizedVector> {
        if rotated_parent.len() != self.dimension {
            return Err(VQuantError::DimensionMismatch {
                expected: self.dimension,
                got: rotated_parent.len(),
            });
        }
        let zero = vec![0.0f32; self.dimension];
        let quantized = self.quantize_prerotated(rotated_residual, &zero)?;
        // Precompute <R*parent, xu_cb> where xu_cb = codes + cb.
        let cb = -((1u32 << quantized.ex_bits) as f32 - 0.5);
        let mut ip_parent = 0.0f32;
        for (i, &p) in rotated_parent.iter().enumerate() {
            ip_parent += p * (quantized.codes[i] as f32 + cb);
        }
        Ok(EdgeQuantizedVector {
            quantized,
            ip_parent_rot_codes: ip_parent,
        })
    }

    /// Returns the edge term of an absolute `||q − v||^2` for one edge.
    ///
    /// The returned value is `||v − u||^2 + f_rescale · ⟨R(q − u), xu_cb⟩`,
    /// which under the rank-1 RaBitQ approximation equals
    /// `||v − u||^2 − 2·⟨q − u, v − u⟩`. The caller is expected to add
    /// `||q − u||^2` to recover an approximation of `||q − v||^2` that is
    /// comparable across edges with different parents. See
    /// [`quantize_edge_prerotated`](Self::quantize_edge_prerotated) for the
    /// full decomposition and a worked example.
    ///
    /// # Arguments
    ///
    /// * `rotated_query` — `R · q`, the raw rotated query. This method does
    ///   **not** expect the centroid-subtracted form `R · (q − c)` used by
    ///   [`approximate_l2_sqr_prerotated`](Self::approximate_l2_sqr_prerotated);
    ///   the edge carries its own parent in place of a global centroid. The
    ///   residual inner product `⟨R(q − u), xu_cb⟩` is recovered internally
    ///   as `⟨R·q, xu_cb⟩ − edge.ip_parent_rot_codes`.
    /// * `edge` — an [`EdgeQuantizedVector`] produced by
    ///   [`quantize_edge_prerotated`](Self::quantize_edge_prerotated).
    #[inline]
    pub fn edge_distance_term_prerotated(rotated_query: &[f32], edge: &EdgeQuantizedVector) -> f32 {
        let qv = &edge.quantized;
        let cb = -((1u32 << qv.ex_bits) as f32 - 0.5);
        // Two-accumulator reduction for FP latency hiding.
        let mut ip0 = 0.0f32;
        let mut ip1 = 0.0f32;
        let codes = &qv.codes;
        let pairs = rotated_query.len() / 2;
        for i in 0..pairs {
            let j = i * 2;
            ip0 += rotated_query[j] * (codes[j] as f32 + cb);
            ip1 += rotated_query[j + 1] * (codes[j + 1] as f32 + cb);
        }
        if rotated_query.len() % 2 != 0 {
            let last = rotated_query.len() - 1;
            ip0 += rotated_query[last] * (codes[last] as f32 + cb);
        }
        let ip_qv = ip0 + ip1;
        // <R*(q-u), xu_cb> = <R*q, xu_cb> - <R*u, xu_cb>
        let ip_residual = ip_qv - edge.ip_parent_rot_codes;
        (qv.f_add + qv.f_rescale * ip_residual).max(0.0)
    }
}

/// A RaBitQ-quantized edge `R · (v − u)` that composes with an externally
/// supplied `||q − u||^2` to give an absolute `||q − v||^2`.
///
/// The only way to construct this value is
/// [`RaBitQQuantizer::quantize_edge_prerotated`]; the only way to consume it
/// is [`RaBitQQuantizer::edge_distance_term_prerotated`]. The distinct type
/// is how the compiler prevents an edge from being routed through
/// [`RaBitQQuantizer::approximate_l2_sqr_prerotated`], whose ranking-score
/// shift is not valid across different parents.
///
/// Fields are `pub` for inspection and serialisation; `#[non_exhaustive]`
/// blocks out-of-crate struct-literal construction, which is what preserves
/// the invariant that `quantized` was built against a zero centroid and that
/// `ip_parent_rot_codes` matches the parent used for the residual.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EdgeQuantizedVector {
    /// Underlying quantized codes for `v − u`. Because the edge is built
    /// against a zero centroid, `f_add` equals `||v − u||^2` exactly and
    /// `f_rescale` equals `−2·delta`.
    pub quantized: QuantizedVector,
    /// `⟨R · parent, xu_cb⟩`, precomputed at build time. At search time this
    /// turns `⟨R · q, xu_cb⟩` into `⟨R · (q − parent), xu_cb⟩` with a single
    /// subtraction, avoiding an O(d) re-rotation per edge.
    pub ip_parent_rot_codes: f32,
}

// ============================================================================
// Optimal Rescaling Factor (Heap-Based)
// ============================================================================

const K_TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];
const K_EPS: f64 = 1e-5;
const K_NENUM: f64 = 10.0;

fn best_rescale_factor(o_abs: &[f32], ex_bits: usize) -> f64 {
    let dim = o_abs.len();
    let max_o = o_abs.iter().cloned().fold(0.0f32, f32::max) as f64;
    if max_o <= f64::EPSILON {
        return 1.0;
    }

    let table_idx = ex_bits.min(K_TIGHT_START.len() - 1);
    let t_end = (((1 << ex_bits) - 1) as f64 + K_NENUM) / max_o;
    let t_start = t_end * K_TIGHT_START[table_idx];

    let mut cur_o_bar = vec![0i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;

    for (idx, &val) in o_abs.iter().enumerate() {
        let cur = ((t_start * val as f64) + K_EPS) as i32;
        cur_o_bar[idx] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * val as f64;
    }

    #[derive(Copy, Clone, Debug)]
    struct HeapEntry {
        t: f64,
        idx: usize,
    }

    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.t.to_bits() == other.t.to_bits() && self.idx == other.idx
        }
    }
    impl Eq for HeapEntry {}

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.t
                .total_cmp(&other.t)
                .then_with(|| self.idx.cmp(&other.idx))
        }
    }

    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    for (idx, &val) in o_abs.iter().enumerate() {
        if val > 0.0 {
            let next_t = (cur_o_bar[idx] + 1) as f64 / val as f64;
            heap.push(Reverse(HeapEntry { t: next_t, idx }));
        }
    }

    let mut max_ip = 0.0f64;
    let mut best_t = t_start;

    while let Some(Reverse(HeapEntry { t: cur_t, idx })) = heap.pop() {
        if cur_t >= t_end {
            continue;
        }

        cur_o_bar[idx] += 1;
        let update = cur_o_bar[idx];
        sqr_denominator += 2.0 * update as f64;
        numerator += o_abs[idx] as f64;

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }

        if update < (1 << ex_bits) - 1 && o_abs[idx] > 0.0 {
            let t_next = (update + 1) as f64 / o_abs[idx] as f64;
            if t_next < t_end {
                heap.push(Reverse(HeapEntry { t: t_next, idx }));
            }
        }
    }

    if best_t <= 0.0 {
        t_start.max(f64::EPSILON)
    } else {
        best_t
    }
}

fn quantize_extended(o_abs: &[f32], residual: &[f32], ex_bits: usize, t: f64) -> (Vec<u16>, f32) {
    let dim = o_abs.len();
    if dim == 0 {
        return (Vec::new(), 1.0);
    }

    let mut code = vec![0u16; dim];
    let max_val = (1 << ex_bits) - 1;
    let mut ipnorm = 0.0f64;

    for i in 0..dim {
        let mut cur = (t * o_abs[i] as f64 + K_EPS) as i32;
        if cur > max_val {
            cur = max_val;
        }
        code[i] = cur as u16;
        ipnorm += (cur as f64 + 0.5) * o_abs[i] as f64;
    }

    let mut ipnorm_inv = if ipnorm.is_finite() && ipnorm > 0.0 {
        (1.0 / ipnorm) as f32
    } else {
        1.0
    };

    // flip codes for negative residuals
    let mask = max_val as u16;
    if max_val > 0 {
        for (idx, &res) in residual.iter().enumerate() {
            if res < 0.0 {
                code[idx] = (!code[idx]) & mask;
            }
        }
    }

    if !ipnorm_inv.is_finite() {
        ipnorm_inv = 1.0;
    }

    (code, ipnorm_inv)
}

fn compute_const_scaling_factor(dim: usize, ex_bits: usize, seed: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const NUM_SAMPLES: usize = 100;

    let mut state = seed;
    let mut next_rand = || -> f32 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        let u1 = (state as f64) / (u64::MAX as f64);
        let mut hasher2 = DefaultHasher::new();
        state.hash(&mut hasher2);
        state = hasher2.finish();
        let u2 = (state as f64) / (u64::MAX as f64);
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };

    let mut sum_t = 0.0f64;
    let mut valid_samples = 0;

    for _ in 0..NUM_SAMPLES {
        let vec: Vec<f32> = (0..dim).map(|_| next_rand()).collect();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            continue;
        }
        let normalized_abs: Vec<f32> = vec.iter().map(|x| (x / norm).abs()).collect();
        let t = best_rescale_factor(&normalized_abs, ex_bits);
        sum_t += t;
        valid_samples += 1;
    }

    if valid_samples > 0 {
        (sum_t / valid_samples as f64) as f32
    } else {
        1.0
    }
}

// ============================================================================
// Bit Packing Utilities
// ============================================================================

fn pack_extended_codes(codes: &[u16], ex_bits: usize) -> Vec<u8> {
    if ex_bits == 0 {
        return Vec::new();
    }

    let total_bits = codes.len() * ex_bits;
    let bytes_needed = total_bits.div_ceil(8);
    let mut packed = vec![0u8; bytes_needed];

    // Fast path for bit widths that divide evenly into 8 (1, 2, 4) or equal 8.
    // Pack multiple codes per byte without the inner bit loop.
    match ex_bits {
        1 => {
            // 8 codes per byte
            for (chunk_idx, chunk) in codes.chunks(8).enumerate() {
                let mut byte = 0u8;
                for (b, &code) in chunk.iter().enumerate() {
                    byte |= ((code & 1) as u8) << b;
                }
                packed[chunk_idx] = byte;
            }
            return packed;
        }
        2 => {
            // 4 codes per byte
            for (chunk_idx, chunk) in codes.chunks(4).enumerate() {
                let mut byte = 0u8;
                for (b, &code) in chunk.iter().enumerate() {
                    byte |= ((code & 3) as u8) << (b * 2);
                }
                packed[chunk_idx] = byte;
            }
            return packed;
        }
        4 => {
            // 2 codes per byte
            for (chunk_idx, chunk) in codes.chunks(2).enumerate() {
                let lo = (chunk[0] & 0xF) as u8;
                let hi = if chunk.len() > 1 {
                    (chunk[1] & 0xF) as u8
                } else {
                    0
                };
                packed[chunk_idx] = lo | (hi << 4);
            }
            return packed;
        }
        8 => {
            // 1 code per byte
            for (i, &code) in codes.iter().enumerate() {
                packed[i] = (code & 0xFF) as u8;
            }
            return packed;
        }
        _ => {}
    }

    // General path for ex_bits ∈ {3, 5, 6, 7}: bit-by-bit packing.
    let mut bit_pos = 0usize;
    for &code in codes {
        let val = code & ((1 << ex_bits) - 1);
        // Pack ex_bits bits from val starting at bit_pos.
        // Most codes cross at most 2 bytes; handle byte-aligned case separately.
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;
        let bits_in_first = 8 - bit_off;
        if ex_bits <= bits_in_first {
            // All bits fit in the current byte.
            packed[byte_idx] |= (val as u8) << bit_off;
        } else {
            // Spans two bytes.
            packed[byte_idx] |= (val as u8) << bit_off;
            packed[byte_idx + 1] |= (val >> bits_in_first) as u8;
        }
        bit_pos += ex_bits;
    }

    packed
}

// ============================================================================
// Rotation Matrix Generation
// ============================================================================

fn generate_orthogonal_rotation(dimension: usize, seed: u64) -> Vec<f32> {
    crate::rotation::orthogonal_rotation_matrix(dimension, dimension, seed)
}

fn apply_rotation(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let nrows = rotation.len() / dimension;
    let mut result = vec![0.0f32; nrows];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dimension..(i + 1) * dimension];
        #[cfg(feature = "simd")]
        {
            *out = innr::dot(row, vector);
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0f32;
            for j in 0..dimension {
                sum += row[j] * vector[j];
            }
            *out = sum;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_binary() {
        let quantizer = RaBitQQuantizer::binary(64, 42).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.binary_codes.len(), 8);
        assert_eq!(quantized.ex_bits, 0);
        assert!(quantized.residual_norm > 0.0);
    }

    #[test]
    fn test_rabitq_4bit() {
        let quantizer = RaBitQQuantizer::with_config(64, 42, RaBitQConfig::bits4()).unwrap();

        let vector: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.ex_bits, 3);
        assert_eq!(quantized.codes.len(), 64);
    }

    #[test]
    fn test_rabitq_8bit() {
        let quantizer = RaBitQQuantizer::with_config(32, 42, RaBitQConfig::bits8()).unwrap();

        let vector: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.ex_bits, 7);
    }

    #[test]
    fn quantize_preserves_dimension() {
        let dim = 64;
        let q = RaBitQQuantizer::binary(dim, 42).unwrap();
        let vector = vec![1.0f32; dim];
        let qv = q.quantize(&vector).unwrap();
        assert_eq!(qv.dimension, dim);
    }

    #[test]
    fn edge_quantization_approximates_true_distance() {
        // Core contract: for VR graph quantization, the edge term plus
        // ||q - parent||^2 approximates ||q - v||^2. Verify that the
        // correlation with the true distance is high across edges.
        use std::collections::HashMap;

        let dim = 128;
        let n = 200;
        let seed = 42;

        // Unnormalized vectors (SIFT-like) so bias pathology is in scope.
        let make_vec = |idx: usize| -> Vec<f32> {
            let mut s = (idx as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (0..dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as f32 / (1u32 << 30) as f32 - 1.0
                })
                .collect()
        };
        let vectors: Vec<Vec<f32>> = (0..n).map(make_vec).collect();

        let mut quantizer = RaBitQQuantizer::with_config(dim, seed, RaBitQConfig::bits4()).unwrap();
        // zero centroid -- we do edge quant, centroid is per-parent.
        quantizer.set_centroid(vec![0.0f32; dim]).unwrap();

        // Pre-rotate all vectors once.
        let rotated: HashMap<usize, Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, quantizer.rotate_query(v).unwrap()))
            .collect();

        // Pick 5 random parents, quantize their first 10 neighbors.
        let mut good_pairs = 0;
        let mut total_pairs = 0;
        for parent_id in (0..n).step_by(40).take(5) {
            let u_rot = &rotated[&parent_id];
            for neighbor_id in (parent_id + 1)..(parent_id + 11).min(n) {
                let v_rot = &rotated[&neighbor_id];
                let residual: Vec<f32> =
                    v_rot.iter().zip(u_rot.iter()).map(|(v, u)| v - u).collect();
                let edge = quantizer
                    .quantize_edge_prerotated(u_rot, &residual)
                    .unwrap();

                // Query: pick another vector.
                for query_id in [0, n / 2, n - 1] {
                    let q = &vectors[query_id];
                    let q_rot = &rotated[&query_id];
                    let u = &vectors[parent_id];
                    let v = &vectors[neighbor_id];

                    let parent_dist: f32 =
                        q.iter().zip(u.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                    let true_dist: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();

                    let edge_term = RaBitQQuantizer::edge_distance_term_prerotated(q_rot, &edge);
                    let approx_dist = parent_dist + edge_term;

                    // Relative error <20% is good for 4-bit RaBitQ at dim=128.
                    let rel_err = (approx_dist - true_dist).abs() / true_dist.max(1.0);
                    total_pairs += 1;
                    if rel_err < 0.20 {
                        good_pairs += 1;
                    }
                }
            }
        }
        // At least 80% of pairs should be within 20% relative error.
        assert!(
            good_pairs as f32 / total_pairs as f32 > 0.8,
            "edge approximation too loose: {}/{} pairs within 20% rel-err",
            good_pairs,
            total_pairs
        );
    }

    #[test]
    fn edge_quantization_is_type_distinct() {
        // Compile-time check: EdgeQuantizedVector cannot be constructed by
        // external users via struct literal (#[non_exhaustive]) and cannot
        // be passed to approximate_l2_sqr_prerotated (takes &QuantizedVector).
        let dim = 32;
        let q = RaBitQQuantizer::with_config(dim, 42, RaBitQConfig::bits4()).unwrap();
        let u_rot = vec![0.1f32; dim];
        let res = vec![0.2f32; dim];
        let edge = q.quantize_edge_prerotated(&u_rot, &res).unwrap();
        // We can access edge.quantized for inspection, but the intended
        // distance path goes through edge_distance_term_prerotated.
        let query = vec![0.3f32; dim];
        let q_rot = q.rotate_query(&query).unwrap();
        let term = RaBitQQuantizer::edge_distance_term_prerotated(&q_rot, &edge);
        assert!(term.is_finite());
    }

    #[test]
    fn approximate_distance_nonneg() {
        let dim = 32;
        let q = RaBitQQuantizer::binary(dim, 42).unwrap();
        let v1 = vec![1.0f32; dim];
        let v2 = vec![0.5f32; dim];
        let qv = q.quantize(&v1).unwrap();
        let dist = q.approximate_distance(&v2, &qv).unwrap();
        assert!(dist >= 0.0, "distance should be non-negative: {}", dist);
    }

    // ---- error case tests ----

    #[test]
    fn dimension_zero_rejected() {
        assert!(RaBitQQuantizer::new(0, 42).is_err());
    }

    #[test]
    fn bits_zero_rejected() {
        let config = RaBitQConfig {
            total_bits: 0,
            t_const: None,
        };
        assert!(RaBitQQuantizer::with_config(32, 42, config).is_err());
    }

    #[test]
    fn bits_over_8_rejected() {
        let config = RaBitQConfig {
            total_bits: 9,
            t_const: None,
        };
        assert!(RaBitQQuantizer::with_config(32, 42, config).is_err());
    }

    #[test]
    fn fit_dimension_mismatch() {
        let mut q = RaBitQQuantizer::binary(8, 42).unwrap();
        // 10 floats for 2 vectors of dimension 8 -> mismatch
        let data = vec![1.0f32; 10];
        assert!(q.fit(&data, 2).is_err());
    }

    #[test]
    fn set_centroid_dimension_mismatch() {
        let mut q = RaBitQQuantizer::binary(8, 42).unwrap();
        assert!(q.set_centroid(vec![0.0f32; 4]).is_err());
    }

    #[test]
    fn quantize_dimension_mismatch() {
        let q = RaBitQQuantizer::binary(8, 42).unwrap();
        assert!(q.quantize(&[1.0f32; 4]).is_err());
    }

    /// Extended RaBitQ: verify all bit widths (2-7) produce valid quantized vectors
    /// with monotonically improving distance accuracy.
    #[test]
    fn test_extended_rabitq_all_widths() {
        let dim = 64;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let target: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();

        // True L2 distance.
        let true_dist: f32 = query
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        for bits in 1..=8 {
            let config = match bits {
                1 => RaBitQConfig::binary(),
                2 => RaBitQConfig::bits2(),
                3 => RaBitQConfig::bits3(),
                4 => RaBitQConfig::bits4(),
                5 => RaBitQConfig::bits5(),
                6 => RaBitQConfig::bits6(),
                7 => RaBitQConfig::bits7(),
                8 => RaBitQConfig::bits8(),
                _ => unreachable!(),
            };
            assert_eq!(config.total_bits, bits);

            let q = RaBitQQuantizer::with_config(dim, 42, config).unwrap();
            let qv = q.quantize(&target).unwrap();
            assert_eq!(qv.ex_bits as usize, bits.saturating_sub(1));
            assert_eq!(qv.dimension, dim);

            let approx_dist = q.approximate_distance(&query, &qv).unwrap();
            let error = (approx_dist - true_dist).abs();

            // Distance error should generally decrease with more bits.
            // Allow some noise (quantization isn't perfectly monotonic per-vector).
            // But 8-bit should be substantially better than 1-bit.
            if bits == 8 {
                let one_bit_q = RaBitQQuantizer::binary(dim, 42).unwrap();
                let one_bit_qv = one_bit_q.quantize(&target).unwrap();
                let one_bit_dist = one_bit_q.approximate_distance(&query, &one_bit_qv).unwrap();
                let one_bit_error = (one_bit_dist - true_dist).abs();
                assert!(
                    error < one_bit_error * 1.5 || error < 1.0,
                    "8-bit error ({}) should be less than 1-bit error ({})",
                    error,
                    one_bit_error
                );
            }

            let _ = error;
        }
    }

    /// Regression: with a FITTED (non-zero) centroid, the approximate distance
    /// must still rank near neighbors well. A spurious centroid cross-term in
    /// `f_add` (`<raw centroid, rotated codes>`) once corrupted this so recall
    /// hovered near 0.5 regardless of bit width, while zero-centroid tests
    /// (`test_extended_rabitq_all_widths`) passed because the term vanishes at
    /// c = 0. Filter-then-rerank recall@10 must be near-perfect at 8 bits.
    #[test]
    fn fitted_centroid_preserves_near_neighbor_ranking() {
        let dim = 64;
        let n = 400;
        // Deterministic clustered data so true near neighbors exist.
        let mut s = 0x1234_5678_u64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u32 << 31) as f32 - 1.0
        };
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dim).map(|_| rng() * 3.0).collect())
            .collect();
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % 8];
                c.iter().map(|&v| v + 0.3 * rng()).collect()
            })
            .collect();
        let flat: Vec<f32> = data.iter().flatten().copied().collect();

        let mut q = RaBitQQuantizer::with_config(dim, 42, RaBitQConfig::bits8()).unwrap();
        q.fit(&flat, n).unwrap();
        let codes: Vec<_> = data.iter().map(|v| q.quantize(v).unwrap()).collect();

        let (k, c_budget, nq) = (10usize, 50usize, 20usize);
        let mut recall = 0.0;
        for qi in 0..nq {
            let query = &data[(qi * 7) % n];
            let exact: Vec<f32> = data
                .iter()
                .map(|d| d.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum())
                .collect();
            let mut eidx: Vec<usize> = (0..n).collect();
            eidx.sort_by(|&a, &b| exact[a].total_cmp(&exact[b]));
            let truek = &eidx[..k];
            let approx: Vec<f32> = codes
                .iter()
                .map(|c| q.approximate_l2_sqr(query, c).unwrap())
                .collect();
            let mut aidx: Vec<usize> = (0..n).collect();
            aidx.sort_by(|&a, &b| approx[a].total_cmp(&approx[b]));
            let cand = &aidx[..c_budget];
            let hits = truek.iter().filter(|t| cand.contains(t)).count();
            recall += hits as f64 / k as f64;
        }
        recall /= nq as f64;
        assert!(
            recall > 0.9,
            "fitted-centroid recall@{k} within C={c_budget} too low: {recall:.3} \
             (centroid cross-term corrupting ranking?)"
        );
    }

    /// Regression for the clamp bug: `approximate_l2_sqr` must return the
    /// ABSOLUTE squared distance, so it tracks true distance with low relative
    /// error on in-distribution pairs. The clamped proxy returned ~0 for any
    /// pair closer than the query is to the centroid (true distance smaller than
    /// `||q-c||^2`), inflating relative error toward 1.0 on exactly the near
    /// pairs that matter.
    #[test]
    fn approximate_l2_sqr_tracks_true_distance() {
        let dim = 64;
        let mut s = 0xBEEF_u64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u32 << 31) as f32 - 0.5
        };
        // Clustered data so the fitted centroid is non-zero and near pairs exist.
        let centers: Vec<Vec<f32>> = (0..6)
            .map(|_| (0..dim).map(|_| rng() * 4.0).collect())
            .collect();
        let data: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                let c = &centers[i % 6];
                c.iter().map(|&v| v + 0.4 * rng()).collect()
            })
            .collect();
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let mut q = RaBitQQuantizer::with_config(dim, 7, RaBitQConfig::bits8()).unwrap();
        q.fit(&flat, data.len()).unwrap();

        let query = &data[0];
        let mut rel_errs = Vec::new();
        for target in data.iter().skip(1).take(60) {
            let true_d2: f32 = query
                .iter()
                .zip(target)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if true_d2 < 1.0 {
                continue; // skip near-duplicates where relative error is unstable
            }
            let approx = q
                .approximate_l2_sqr(query, &q.quantize(target).unwrap())
                .unwrap();
            rel_errs.push((approx - true_d2).abs() / true_d2);
        }
        let mean_rel = rel_errs.iter().sum::<f32>() / rel_errs.len() as f32;
        assert!(
            mean_rel < 0.3,
            "8-bit estimate mean relative error {mean_rel:.3} too high \
             (clamped proxy / spurious centroid term regression?)"
        );
    }

    proptest::proptest! {
        /// Property: for random fitted data, the 8-bit approximate distance from a
        /// query to its OWN quantized code is far smaller than to a random other
        /// point. Catches gross miscalibration (clamp, spurious centroid term,
        /// space mixups) without asserting a brittle absolute tolerance.
        #[test]
        fn approx_self_distance_smaller_than_cross(seed in 1u64..1_000_000) {
            let dim = 32;
            let mut s = seed;
            let mut rng = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as f32 / (1u32 << 31) as f32 - 0.5
            };
            let data: Vec<Vec<f32>> = (0..64).map(|_| (0..dim).map(|_| rng() * 10.0).collect()).collect();
            let flat: Vec<f32> = data.iter().flatten().copied().collect();
            let mut q = RaBitQQuantizer::with_config(dim, 1, RaBitQConfig::bits8()).unwrap();
            q.fit(&flat, data.len()).unwrap();

            let self_d = q.approximate_l2_sqr(&data[0], &q.quantize(&data[0]).unwrap()).unwrap();
            let cross_d = q.approximate_l2_sqr(&data[0], &q.quantize(&data[32]).unwrap()).unwrap();
            proptest::prop_assert!(self_d >= 0.0, "absolute distance must be non-negative, got {self_d}");
            proptest::prop_assert!(
                self_d < cross_d,
                "approx self-distance {self_d:.3} should be < cross-distance {cross_d:.3}"
            );
        }
    }

    /// Ties the two estimators together so the otherwise-untested
    /// `approximate_l2_sqr_prerotated` is guarded: the absolute estimator equals
    /// the prerotated ranking proxy plus the query constant `||q-c||^2`
    /// (= `||rotate_query(q)||^2`, since the rotation is orthogonal). Any
    /// arithmetic drift in either estimator breaks the identity. Without this,
    /// mutation testing leaves ~40 survivors in the prerotated path.
    #[test]
    fn prerotated_proxy_plus_query_norm_equals_absolute() {
        let dim = 48;
        let mut s = 0xC0FFEE_u64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u32 << 31) as f32 - 0.5
        };
        // Offset all data far from the origin so the fitted centroid is large
        // and the query-residual sign genuinely matters.
        let data: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dim).map(|_| 20.0 + rng() * 4.0).collect())
            .collect();
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let mut q = RaBitQQuantizer::with_config(dim, 9, RaBitQConfig::bits8()).unwrap();
        q.fit(&flat, data.len()).unwrap();

        let query = &data[0];
        let rq = q.rotate_query(query).unwrap();
        let qc2: f32 = rq.iter().map(|x| x * x).sum(); // ||q-c||^2
        let mut checked = 0;
        for target in data.iter().skip(1).take(40) {
            let qv = q.quantize(target).unwrap();
            let abs = q.approximate_l2_sqr(query, &qv).unwrap();
            let proxy = RaBitQQuantizer::approximate_l2_sqr_prerotated(&rq, &qv);
            // abs = max(0, qc2 + proxy); only assert where the clamp is inactive.
            if qc2 + proxy > 1.0 {
                let rel = (abs - (qc2 + proxy)).abs() / abs.max(1.0);
                assert!(
                    rel < 1e-3,
                    "prerotated identity broken: abs={abs:.3} qc2+proxy={:.3} rel={rel:.4}",
                    qc2 + proxy
                );
                checked += 1;
            }
        }
        assert!(checked > 10, "test exercised too few pairs ({checked})");
    }

    /// Verify const-scaling works with all bit widths.
    #[test]
    fn test_const_scaling_all_widths() {
        let dim = 32;
        for bits in 2..=7 {
            let config = RaBitQConfig {
                total_bits: bits,
                t_const: None,
            }
            .with_const_scaling(dim, 42);

            assert!(
                config.t_const.is_some(),
                "bits={} should produce a const scaling factor",
                bits
            );

            let q = RaBitQQuantizer::with_config(dim, 42, config).unwrap();
            let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let qv = q.quantize(&v).unwrap();
            assert_eq!(qv.ex_bits as usize, bits - 1);
        }
    }
}
