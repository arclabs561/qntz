//! RaBitQ (Randomized Binary Quantization).
//!
//! This module is extracted from `jin` as a low-level, reusable primitive.

use crate::VQuantError;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Configuration for RaBitQ quantization.
#[derive(Clone, Copy, Debug)]
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

    /// 4-bit quantization (default, good balance).
    #[must_use]
    pub fn bits4() -> Self {
        Self {
            total_bits: 4,
            t_const: None,
        }
    }

    /// 8-bit quantization (high accuracy).
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
        let (f_add, f_rescale, f_error, residual_norm) =
            self.compute_correction_factors(&rotated, centroid, &binary_codes_unpacked);

        // Step 7: delta/vl
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let quantized_shifted: Vec<f32> =
            total_codes.iter().map(|&code| code as f32 + cb).collect();

        let norm_quan_sqr: f32 = quantized_shifted.iter().map(|x| x * x).sum();
        let norm_residual_sqr: f32 = rotated.iter().map(|x| x * x).sum();
        let dot_rq: f32 = rotated
            .iter()
            .zip(quantized_shifted.iter())
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
        let binary_codes = pack_binary_codes(&binary_codes_unpacked);
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

    fn compute_correction_factors(
        &self,
        residual: &[f32],
        centroid: &[f32],
        binary_codes: &[u8],
    ) -> (f32, f32, f32, f32) {
        let dim = self.dimension;

        // centered binary codes
        let xu_cb: Vec<f32> = binary_codes.iter().map(|&bit| bit as f32 - 0.5).collect();

        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();
        let xu_cb_norm_sqr: f32 = xu_cb.iter().map(|x| x * x).sum();
        let ip_resi_xucb: f32 = residual.iter().zip(xu_cb.iter()).map(|(r, x)| r * x).sum();
        let ip_cent_xucb: f32 = centroid.iter().zip(xu_cb.iter()).map(|(c, x)| c * x).sum();

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

        let f_add = l2_sqr + 2.0 * l2_sqr * ip_cent_xucb / denom;
        let f_rescale = -2.0 * l2_sqr / denom;
        let f_error = 2.0 * tmp_error;

        (f_add, f_rescale, f_error, l2_norm)
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

        let dist = quantized.f_add + quantized.f_rescale * ip;
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

fn pack_binary_codes(codes: &[u8]) -> Vec<u8> {
    let bytes_needed = codes.len().div_ceil(8);
    let mut packed = vec![0u8; bytes_needed];
    for (i, &code) in codes.iter().enumerate() {
        if code != 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

fn pack_extended_codes(codes: &[u16], ex_bits: usize) -> Vec<u8> {
    if ex_bits == 0 {
        return Vec::new();
    }

    let total_bits = codes.len() * ex_bits;
    let bytes_needed = total_bits.div_ceil(8);
    let mut packed = vec![0u8; bytes_needed];

    let mut bit_pos = 0;
    for &code in codes {
        let val = code & ((1 << ex_bits) - 1);
        for b in 0..ex_bits {
            if (val >> b) & 1 != 0 {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                if byte_idx < packed.len() {
                    packed[byte_idx] |= 1 << bit_idx;
                }
            }
            bit_pos += 1;
        }
    }

    packed
}

// ============================================================================
// Rotation Matrix Generation
// ============================================================================

fn generate_orthogonal_rotation(dimension: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut rotation = vec![0.0f32; dimension * dimension];

    let mut state = seed;
    let mut next_rand = || -> f32 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
    };

    let mut basis: Vec<Vec<f32>> = Vec::new();

    for i in 0..dimension {
        let mut v: Vec<f32> = (0..dimension).map(|_| next_rand()).collect();

        // orthogonalize
        for b in &basis {
            let dot: f32 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }

        // normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
            basis.push(v);
        } else {
            let mut v = vec![0.0f32; dimension];
            v[i] = 1.0;
            basis.push(v);
        }
    }

    for (i, row) in basis.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            rotation[i * dimension + j] = val;
        }
    }

    rotation
}

fn apply_rotation(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dimension];
    for (i, out) in result.iter_mut().enumerate() {
        let row_start = i * dimension;
        let mut sum = 0.0;
        for j in 0..dimension {
            sum += rotation[row_start + j] * vector[j];
        }
        *out = sum;
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
}
