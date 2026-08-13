//! Distributional quantization with codebooks fitted to distributional priors.
//!
//! Given a distributional assumption about the data (Gaussian, logistic, Gumbel, etc.),
//! distributional quantization places reconstruction points according to quantiles of the
//! selected distribution.
//!
//! This complements RaBitQ (hardware-friendly ANN codes) when a distributional
//! assumption is useful.
//!
//! # References
//!
//! - Petersen & Sutter, "Distributional Quantization" (2021-2023)
//! - Lloyd (1982), "Least squares quantization in PCM"
//!
//! # Example
//!
//! ```rust
//! use qntz::distquant::{quantize, dequantize, Distribution};
//!
//! let data: Vec<f64> = vec![0.1, -0.5, 1.2, -0.3, 0.7, -1.1, 0.4, 0.9];
//!
//! let (codes, info) = quantize(&data, 4, Distribution::Gaussian, None);
//! let reconstructed = dequantize(&codes, &info);
//!
//! // Reconstruction error is bounded by the quantization resolution
//! for (orig, recon) in data.iter().zip(reconstructed.iter()) {
//!     assert!((orig - recon).abs() < 1.0);
//! }
//! ```

use crate::{Result, VQuantError};

/// Distributional assumption for quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Standard normal distribution. Best for embeddings, weights.
    Gaussian,
    /// Logistic distribution. Heavier tails than Gaussian.
    Logistic,
    /// Gumbel (max) distribution. Asymmetric, right-skewed.
    Gumbel,
    /// Cauchy distribution. Very heavy tails.
    Cauchy,
    /// Exponential distribution (positive values only).
    Exponential,
}

/// Metadata needed for dequantization.
#[derive(Debug, Clone)]
pub struct QuantInfo {
    /// The codebook (reconstruction points).
    pub codebook: Vec<f64>,
    /// Location (shift) applied before quantization.
    pub loc: f64,
    /// Scale applied before quantization.
    pub scale: f64,
    /// Number of bits per value.
    pub bits: u8,
    /// Distribution used.
    pub dist: Distribution,
}

/// Quantile function (inverse CDF) for each distribution.
fn quantile(dist: Distribution, p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    match dist {
        Distribution::Gaussian => gaussian_quantile(p),
        Distribution::Logistic => {
            // Q(p) = ln(p / (1-p))
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }
            (p / (1.0 - p)).ln()
        }
        Distribution::Gumbel => {
            // Q(p) = -ln(-ln(p))  (standard Gumbel-max)
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }
            -(-p.ln()).ln()
        }
        Distribution::Cauchy => {
            // Q(p) = tan(pi * (p - 0.5))
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }
            (std::f64::consts::PI * (p - 0.5)).tan()
        }
        Distribution::Exponential => {
            // Q(p) = -ln(1-p)  (rate=1)
            if p >= 1.0 {
                return f64::INFINITY;
            }
            -(1.0 - p).ln()
        }
    }
}

/// CDF for each distribution.
#[cfg(test)]
fn cdf(dist: Distribution, x: f64) -> f64 {
    match dist {
        Distribution::Gaussian => gaussian_cdf(x),
        Distribution::Logistic => 1.0 / (1.0 + (-x).exp()),
        Distribution::Gumbel => (-(-x).exp()).exp(),
        Distribution::Cauchy => 0.5 + (x).atan() / std::f64::consts::PI,
        Distribution::Exponential => {
            if x < 0.0 {
                0.0
            } else {
                1.0 - (-x).exp()
            }
        }
    }
}

/// Rational approximation to the Gaussian quantile (Abramowitz & Stegun).
fn gaussian_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Rational approximation (Peter Acklam's algorithm)
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Gaussian CDF via error function approximation.
#[cfg(test)]
fn gaussian_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
#[cfg(test)]
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = 0.254_829_592 * t - 0.284_496_736 * t2 + 1.421_413_741 * t3 - 1.453_152_027 * t4
        + 1.061_405_429 * t5;

    sign * (1.0 - poly * (-x * x).exp())
}

/// Build a quantile-based codebook for `n` levels under the given distribution.
///
/// Places reconstruction points at the quantile function evaluated at evenly spaced
/// probabilities.
///
/// The `beta` parameter scales the range: larger beta covers more of the distribution's
/// tails. `None` uses 1.0 (standard).
pub fn build_codebook(n: usize, dist: Distribution, beta: Option<f64>) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(VQuantError::InvalidConfig {
            field: "n",
            reason: "number of quantization levels must be > 0",
        });
    }

    let beta = beta.unwrap_or(1.0);
    if beta <= 0.0 || !beta.is_finite() {
        return Err(VQuantError::InvalidConfig {
            field: "beta",
            reason: "beta must be positive and finite",
        });
    }

    let mut codebook = Vec::with_capacity(n);

    for i in 0..n {
        // Midpoint of the i-th quantile bin
        let p_lo = (i as f64) / (n as f64);
        let p_hi = ((i + 1) as f64) / (n as f64);
        let p_mid = (p_lo + p_hi) / 2.0;

        // Clamp away from 0 and 1 to avoid infinities
        let p_clamped = p_mid.clamp(1e-6, 1.0 - 1e-6);
        codebook.push(quantile(dist, p_clamped) * beta);
    }

    Ok(codebook)
}

/// Find the nearest codebook index for a standardized value.
fn nearest_code(value: f64, codebook: &[f64]) -> u32 {
    let mut best = 0u32;
    let mut best_dist = f64::INFINITY;
    for (i, &c) in codebook.iter().enumerate() {
        let d = (value - c).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u32;
        }
    }
    best
}

/// Quantize a slice of f64 values.
///
/// # Arguments
///
/// * `data` - Input values
/// * `bits` - Number of bits per value (1-8, giving 2-256 levels)
/// * `dist` - Distributional assumption
/// * `beta` - Scale factor for the codebook range (`None` = auto-search)
///
/// # Returns
///
/// A tuple of `(codes, info)` where codes are indices into the codebook.
///
/// # Panics
///
/// Panics when `bits` is outside `1..=8` or an explicit `beta` is not positive
/// and finite. Use [`try_quantize`] to handle invalid configuration.
pub fn quantize(
    data: &[f64],
    bits: u8,
    dist: Distribution,
    beta: Option<f64>,
) -> (Vec<u32>, QuantInfo) {
    try_quantize(data, bits, dist, beta)
        .expect("bits must be in 1..=8 and beta must be positive and finite")
}

/// Checked form of [`quantize`].
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] when `bits` is outside `1..=8` or
/// an explicit `beta` is not positive and finite.
pub fn try_quantize(
    data: &[f64],
    bits: u8,
    dist: Distribution,
    beta: Option<f64>,
) -> Result<(Vec<u32>, QuantInfo)> {
    if !(1..=8).contains(&bits) {
        return Err(VQuantError::InvalidConfig {
            field: "bits",
            reason: "bits must be in 1..=8",
        });
    }
    if beta.is_some_and(|value| value <= 0.0 || !value.is_finite()) {
        return Err(VQuantError::InvalidConfig {
            field: "beta",
            reason: "beta must be positive and finite",
        });
    }

    if data.is_empty() {
        return Ok((
            vec![],
            QuantInfo {
                codebook: vec![],
                loc: 0.0,
                scale: 1.0,
                bits,
                dist,
            },
        ));
    }

    let n_levels = 1usize << bits;

    // Standardize: shift to zero mean, unit variance
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = var.sqrt().max(1e-12);

    // Auto-search beta if not provided
    let beta = beta.unwrap_or_else(|| find_optimal_beta(data, mean, std, n_levels, dist));

    let codebook = build_codebook(n_levels, dist, Some(beta))?;

    // Quantize: map each standardized value to nearest codebook entry
    let codes: Vec<u32> = data
        .iter()
        .map(|&x| {
            let standardized = (x - mean) / std;
            nearest_code(standardized, &codebook)
        })
        .collect();

    Ok((
        codes,
        QuantInfo {
            codebook,
            loc: mean,
            scale: std,
            bits,
            dist,
        },
    ))
}

/// Dequantize codes back to f64 values.
pub fn dequantize(codes: &[u32], info: &QuantInfo) -> Vec<f64> {
    codes
        .iter()
        .map(|&c| {
            let cb_val = info.codebook.get(c as usize).copied().unwrap_or(0.0);
            cb_val * info.scale + info.loc
        })
        .collect()
}

/// Pack quantized codes into u64 words for compact storage.
///
/// For `k` bits per code, packs `floor(64/k)` codes per u64.
/// Returns an empty vector for an invalid bit width; use [`try_pack_codes`] to
/// distinguish invalid configuration from empty input.
pub fn pack_codes(codes: &[u32], bits: u8) -> Vec<u64> {
    try_pack_codes(codes, bits).unwrap_or_default()
}

/// Checked form of [`pack_codes`].
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] unless `bits` is in `1..=32`.
pub fn try_pack_codes(codes: &[u32], bits: u8) -> Result<Vec<u64>> {
    validate_code_bits(bits)?;
    if codes.is_empty() {
        return Ok(vec![]);
    }

    let codes_per_word = 64 / bits as usize;
    let n_words = codes.len().div_ceil(codes_per_word);
    let mask = (1u64 << u32::from(bits)) - 1;

    let mut packed = vec![0u64; n_words];
    for (i, &code) in codes.iter().enumerate() {
        let word_idx = i / codes_per_word;
        let bit_offset = (i % codes_per_word) * bits as usize;
        packed[word_idx] |= ((code as u64) & mask) << bit_offset;
    }
    Ok(packed)
}

/// Unpack codes from u64 words.
///
/// Returns an empty vector for an invalid width or undersized input; use
/// [`try_unpack_codes`] to distinguish those cases from empty output.
pub fn unpack_codes(packed: &[u64], bits: u8, n_codes: usize) -> Vec<u32> {
    try_unpack_codes(packed, bits, n_codes).unwrap_or_default()
}

/// Checked form of [`unpack_codes`].
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] unless `bits` is in `1..=32`, or
/// [`VQuantError::DimensionMismatch`] when `packed` cannot contain `n_codes`.
pub fn try_unpack_codes(packed: &[u64], bits: u8, n_codes: usize) -> Result<Vec<u32>> {
    validate_code_bits(bits)?;
    if n_codes == 0 {
        return Ok(vec![]);
    }

    let codes_per_word = 64 / bits as usize;
    let required = n_codes.div_ceil(codes_per_word);
    if packed.len() < required {
        return Err(VQuantError::DimensionMismatch {
            expected: required,
            got: packed.len(),
        });
    }
    let mask = (1u64 << u32::from(bits)) - 1;

    let mut codes = Vec::with_capacity(n_codes);
    for i in 0..n_codes {
        let word_idx = i / codes_per_word;
        let bit_offset = (i % codes_per_word) * bits as usize;
        codes.push(((packed[word_idx] >> bit_offset) & mask) as u32);
    }
    Ok(codes)
}

fn validate_code_bits(bits: u8) -> Result<()> {
    if !(1..=32).contains(&bits) {
        return Err(VQuantError::InvalidConfig {
            field: "bits",
            reason: "bits must be in 1..=32 for u32 codes",
        });
    }
    Ok(())
}

/// Search for the beta that minimizes mean squared reconstruction error.
fn find_optimal_beta(
    data: &[f64],
    mean: f64,
    std: f64,
    n_levels: usize,
    dist: Distribution,
) -> f64 {
    let mut best_beta = 1.0;
    let mut best_mse = f64::INFINITY;

    // Grid search over beta values
    for i in 1..=20 {
        let beta = i as f64 * 0.25;
        let codebook = match build_codebook(n_levels, dist, Some(beta)) {
            Ok(cb) => cb,
            Err(_) => continue,
        };

        let mse: f64 = data
            .iter()
            .map(|&x| {
                let z = (x - mean) / std;
                let code = nearest_code(z, &codebook);
                let recon = codebook[code as usize];
                (z - recon).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64;

        if mse < best_mse {
            best_mse = mse;
            best_beta = beta;
        }
    }

    best_beta
}

/// Compute the mean squared error of quantization for analysis.
pub fn quantization_mse(data: &[f64], codes: &[u32], info: &QuantInfo) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let reconstructed = dequantize(codes, info);
    data.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / data.len() as f64
}

/// Find the best distribution for the given data.
///
/// Tests all distributions and returns the one with lowest MSE.
pub fn best_distribution(data: &[f64], bits: u8) -> (Distribution, f64) {
    let candidates = [
        Distribution::Gaussian,
        Distribution::Logistic,
        Distribution::Gumbel,
        Distribution::Cauchy,
        Distribution::Exponential,
    ];

    let mut best_dist = Distribution::Gaussian;
    let mut best_mse = f64::INFINITY;

    for &dist in &candidates {
        let (codes, info) = quantize(data, bits, dist, None);
        let mse = quantization_mse(data, &codes, &info);
        if mse < best_mse {
            best_mse = mse;
            best_dist = dist;
        }
    }

    (best_dist, best_mse)
}

// ── Lloyd-Max optimal codebooks ─────────────────────────────────────────────
//
// The Lloyd-Max iteration refines reconstruction points and decision boundaries
// toward a local stationary point of expected MSE. This is the classical
// iterative approach (as opposed to the quantile-placement heuristic above).
// For symmetric distributions the codebook is symmetric by construction.
//
// References:
// - Lloyd (1982). "Least squares quantization in PCM."
// - Max (1960). "Quantizing for minimum distortion."

/// Distribution family for Lloyd-Max codebook computation.
///
/// These are standard-form (location 0, scale 1) distributions with known
/// closed-form PDF, CDF, and first-moment antiderivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LloydMaxDist {
    /// Standard Gaussian N(0,1).
    Gaussian,
    /// Standard logistic distribution (mean 0, scale 1).
    Logistic,
    /// Standard Cauchy distribution (location 0, scale 1).
    /// Outer bins are truncated at a finite radius (no finite mean).
    Cauchy,
}

impl LloydMaxDist {
    /// Probability density function.
    fn pdf(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => (-0.5 * x * x).exp() / (2.0 * PI).sqrt(),
            Self::Logistic => {
                let e = (-x).exp();
                e / (1.0 + e).powi(2)
            }
            Self::Cauchy => 1.0 / (PI * (1.0 + x * x)),
        }
    }

    /// Cumulative distribution function.
    fn lloyd_cdf(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => 0.5 * erfc_approx(-x * FRAC_1_SQRT_2),
            Self::Logistic => 1.0 / (1.0 + (-x).exp()),
            Self::Cauchy => 0.5 + x.atan() / PI,
        }
    }

    /// Whether the outer bins should use finite truncation.
    fn needs_truncation(self) -> bool {
        matches!(self, Self::Cauchy)
    }

    /// Truncation radius for distributions without finite mean.
    fn truncation_radius(self) -> f64 {
        match self {
            Self::Cauchy => 200.0,
            _ => unreachable!(),
        }
    }

    /// Effective support radius for boundary initialization.
    fn init_radius(self) -> f64 {
        match self {
            Self::Gaussian => 4.0,
            Self::Logistic => 8.0,
            Self::Cauchy => 50.0,
        }
    }
}

/// Result of Lloyd-Max codebook computation.
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    /// Decision boundaries, length `levels - 1`.
    /// `boundaries[i]` is the threshold between level `i` and level `i+1`.
    pub boundaries: Vec<f64>,
    /// Representative (reconstruction) values, length `levels`.
    pub representatives: Vec<f64>,
    /// Number of Lloyd-Max iterations until convergence.
    pub iterations: usize,
}

/// Compute a quantization codebook for a symmetric distribution using the
/// Lloyd-Max iteration.
///
/// Returns a [`LloydMaxCodebook`] with `levels` reconstruction points and
/// `levels - 1` decision boundaries.
///
/// # Errors
///
/// Returns [`VQuantError::InvalidConfig`] if `levels` is zero.
///
/// # Examples
///
/// ```
/// use qntz::distquant::{lloyd_max_codebook, LloydMaxDist};
///
/// let cb = lloyd_max_codebook(LloydMaxDist::Gaussian, 4).unwrap();
/// assert_eq!(cb.representatives.len(), 4);
/// assert_eq!(cb.boundaries.len(), 3);
///
/// // Boundaries are sorted.
/// for w in cb.boundaries.windows(2) {
///     assert!(w[0] < w[1]);
/// }
/// ```
pub fn lloyd_max_codebook(dist: LloydMaxDist, levels: usize) -> Result<LloydMaxCodebook> {
    if levels == 0 {
        return Err(VQuantError::InvalidConfig {
            field: "levels",
            reason: "number of quantization levels must be > 0",
        });
    }

    if levels == 1 {
        return Ok(LloydMaxCodebook {
            boundaries: vec![],
            representatives: vec![0.0],
            iterations: 0,
        });
    }

    let n = levels;
    let even = n % 2 == 0;
    let h = n / 2;

    let truncated = dist.needs_truncation();
    let outer = if truncated {
        dist.truncation_radius()
    } else {
        f64::INFINITY
    };

    let r = dist.init_radius();

    let nb = if even { h - 1 } else { h };

    let mut pos_b: Vec<f64> = (0..nb)
        .map(|i| r * (i as f64 + 1.0) / (nb as f64 + 1.0))
        .collect();

    let max_iter = 1000;
    let tol = 1e-8;
    let mut pos_r = vec![0.0; h];
    let mut iters = 0;

    for iter in 0..max_iter {
        iters = iter + 1;

        let left_edge = if even { 0.0 } else { pos_b[0] };
        let mut full_b = Vec::with_capacity(h + 1);
        full_b.push(left_edge);
        if even {
            full_b.extend_from_slice(&pos_b);
        } else {
            full_b.extend_from_slice(&pos_b[1..]);
        }
        full_b.push(outer);

        for i in 0..h {
            pos_r[i] = lloyd_conditional_mean(dist, full_b[i], full_b[i + 1]);
        }

        let mut max_delta = 0.0_f64;

        if even {
            for i in 0..nb {
                let new_b = 0.5 * (pos_r[i] + pos_r[i + 1]);
                max_delta = max_delta.max((new_b - pos_b[i]).abs());
                pos_b[i] = new_b;
            }
        } else {
            let new_b0 = pos_r[0] / 2.0;
            max_delta = max_delta.max((new_b0 - pos_b[0]).abs());
            pos_b[0] = new_b0;
            for i in 1..nb {
                let new_b = 0.5 * (pos_r[i - 1] + pos_r[i]);
                max_delta = max_delta.max((new_b - pos_b[i]).abs());
                pos_b[i] = new_b;
            }
        }

        if max_delta < tol {
            break;
        }
    }

    let mut representatives = Vec::with_capacity(n);
    let mut boundaries = Vec::with_capacity(n - 1);

    for i in (0..h).rev() {
        representatives.push(-pos_r[i]);
    }
    if !even {
        representatives.push(0.0);
    }
    for r in pos_r.iter().take(h) {
        representatives.push(*r);
    }

    if even {
        for i in (0..nb).rev() {
            boundaries.push(-pos_b[i]);
        }
        boundaries.push(0.0);
        for b in pos_b.iter().take(nb) {
            boundaries.push(*b);
        }
    } else {
        for i in (0..nb).rev() {
            boundaries.push(-pos_b[i]);
        }
        for b in pos_b.iter().take(nb) {
            boundaries.push(*b);
        }
    }

    Ok(LloydMaxCodebook {
        boundaries,
        representatives,
        iterations: iters,
    })
}

/// Map a continuous value to the index of its quantization level.
///
/// Uses binary search on `boundaries` (length `levels - 1`).
pub fn lloyd_max_quantize(value: f64, boundaries: &[f64]) -> usize {
    boundaries.partition_point(|&b| b <= value)
}

/// Map a quantization level index back to its representative value.
///
/// # Panics
///
/// Panics if `level >= representatives.len()`.
pub fn lloyd_max_dequantize(level: usize, representatives: &[f64]) -> f64 {
    representatives[level]
}

// ── Lloyd-Max numerical helpers ────────────────────────────────────────────

use core::f64::consts::{FRAC_1_SQRT_2, PI};

/// Complementary error function (Abramowitz & Stegun 7.1.26). Accuracy ~1.5e-7.
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Conditional mean `E[X | lo <= X < hi]` for a given distribution.
fn lloyd_conditional_mean(dist: LloydMaxDist, lo: f64, hi: f64) -> f64 {
    let prob = dist.lloyd_cdf(hi) - dist.lloyd_cdf(lo);
    if prob < 1e-300 {
        return match (lo.is_finite(), hi.is_finite()) {
            (true, true) => 0.5 * (lo + hi),
            (false, true) => hi,
            (true, false) => lo,
            (false, false) => 0.0,
        };
    }
    let num = lloyd_first_moment_antideriv(dist, hi) - lloyd_first_moment_antideriv(dist, lo);
    num / prob
}

/// Antiderivative of `x * f(x)`, evaluated at `x`.
fn lloyd_first_moment_antideriv(dist: LloydMaxDist, x: f64) -> f64 {
    match dist {
        LloydMaxDist::Gaussian => {
            if !x.is_finite() {
                return 0.0;
            }
            -dist.pdf(x)
        }
        LloydMaxDist::Logistic => {
            if !x.is_finite() {
                return 0.0;
            }
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let softplus = if x > 20.0 {
                x + (-x).exp()
            } else if x < -20.0 {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            x * sigmoid - softplus
        }
        LloydMaxDist::Cauchy => {
            debug_assert!(x.is_finite(), "Cauchy F1 must use finite bounds");
            (1.0 + x * x).ln() / (2.0 * PI)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_quantile_symmetry() {
        let q25 = gaussian_quantile(0.25);
        let q75 = gaussian_quantile(0.75);
        assert!(
            (q25 + q75).abs() < 1e-10,
            "Gaussian quantile should be symmetric"
        );
    }

    #[test]
    fn test_gaussian_quantile_median() {
        let q50 = gaussian_quantile(0.5);
        assert!(q50.abs() < 1e-10, "Median of Gaussian should be 0");
    }

    #[test]
    fn test_codebook_monotonic() {
        for dist in [
            Distribution::Gaussian,
            Distribution::Logistic,
            Distribution::Gumbel,
            Distribution::Cauchy,
        ] {
            let cb = build_codebook(16, dist, Some(1.0)).unwrap();
            for i in 1..cb.len() {
                assert!(
                    cb[i] >= cb[i - 1],
                    "Codebook not monotonic for {:?} at index {}",
                    dist,
                    i
                );
            }
        }
    }

    #[test]
    fn test_exponential_codebook_positive() {
        let cb = build_codebook(8, Distribution::Exponential, Some(1.0)).unwrap();
        for &v in &cb {
            assert!(v >= 0.0, "Exponential codebook should be non-negative");
        }
    }

    #[test]
    fn test_roundtrip_gaussian() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 20.0).collect();
        let (codes, info) = quantize(&data, 4, Distribution::Gaussian, None);
        let reconstructed = dequantize(&codes, &info);

        assert_eq!(data.len(), reconstructed.len());
        let mse = quantization_mse(&data, &codes, &info);
        assert!(mse < 0.1, "MSE too high: {mse}");
    }

    #[test]
    fn test_roundtrip_logistic() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 20.0).collect();
        let (codes, info) = quantize(&data, 4, Distribution::Logistic, None);
        let reconstructed = dequantize(&codes, &info);

        assert_eq!(data.len(), reconstructed.len());
        let mse = quantization_mse(&data, &codes, &info);
        assert!(mse < 0.1, "MSE too high: {mse}");
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let codes: Vec<u32> = (0..20).map(|i| i % 16).collect();
        let packed = pack_codes(&codes, 4);
        let unpacked = unpack_codes(&packed, 4, codes.len());
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_3bit() {
        let codes: Vec<u32> = (0..30).map(|i| i % 8).collect();
        let packed = pack_codes(&codes, 3);
        let unpacked = unpack_codes(&packed, 3, codes.len());
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn checked_quantize_rejects_invalid_bits_and_beta() {
        let data = [0.0, 1.0];
        for bits in [0, 9, u8::MAX] {
            assert!(try_quantize(&data, bits, Distribution::Gaussian, None).is_err());
        }
        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(try_quantize(&data, 4, Distribution::Gaussian, Some(beta)).is_err());
        }
    }

    #[test]
    fn checked_code_packing_matches_independent_slot_oracle() {
        for bits in 1..=32u8 {
            let max = if bits == 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let codes: Vec<u32> = (0..137)
                .map(|i| match i % 5 {
                    0 => 0,
                    1 => 1,
                    2 => max,
                    3 => max / 2,
                    _ => (i as u32).wrapping_mul(0x9e37_79b9) & max,
                })
                .collect();
            let packed = try_pack_codes(&codes, bits).unwrap();

            let per_word = 64 / bits as usize;
            for (i, &code) in codes.iter().enumerate() {
                let mut oracle = 0u32;
                let base = (i % per_word) * bits as usize;
                for bit in 0..bits as usize {
                    let set = (packed[i / per_word] >> (base + bit)) & 1;
                    oracle |= (set as u32) << bit;
                }
                assert_eq!(oracle, code, "independent decode failed at width {bits}");
            }
            assert_eq!(try_unpack_codes(&packed, bits, codes.len()).unwrap(), codes);
        }
    }

    #[test]
    fn checked_code_packing_rejects_invalid_widths_and_short_input() {
        for bits in [0, 33, 64, u8::MAX] {
            assert!(try_pack_codes(&[1], bits).is_err());
            assert!(try_unpack_codes(&[1], bits, 1).is_err());
        }
        assert!(try_unpack_codes(&[], 4, 1).is_err());
        assert!(try_unpack_codes(&[0], 4, 17).is_err());
    }

    #[test]
    fn test_empty_data() {
        let (codes, info) = quantize(&[], 4, Distribution::Gaussian, None);
        assert!(codes.is_empty());
        let recon = dequantize(&codes, &info);
        assert!(recon.is_empty());
    }

    #[test]
    fn test_single_value() {
        let data = [42.0];
        let (codes, info) = quantize(&data, 4, Distribution::Gaussian, None);
        let recon = dequantize(&codes, &info);
        // Single value: std is clamped to 1e-12, so reconstruction should be close
        assert_eq!(recon.len(), 1);
    }

    #[test]
    fn test_best_distribution() {
        // Roughly Gaussian data
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 200.0).collect();
        let (dist, mse) = best_distribution(&data, 4);
        assert!(mse < 0.05, "Best distribution MSE too high: {mse}");
        // Uniform-ish data, but the test is that it runs without panicking
        let _ = dist;
    }

    #[test]
    fn test_more_bits_lower_mse() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 40.0).collect();
        let (codes2, info2) = quantize(&data, 2, Distribution::Gaussian, None);
        let (codes4, info4) = quantize(&data, 4, Distribution::Gaussian, None);
        let (codes8, info8) = quantize(&data, 8, Distribution::Gaussian, None);

        let mse2 = quantization_mse(&data, &codes2, &info2);
        let mse4 = quantization_mse(&data, &codes4, &info4);
        let mse8 = quantization_mse(&data, &codes8, &info8);

        assert!(mse4 < mse2, "4-bit should beat 2-bit: {mse4} vs {mse2}");
        assert!(mse8 < mse4, "8-bit should beat 4-bit: {mse8} vs {mse4}");
    }

    #[test]
    fn test_cdf_quantile_inverse() {
        for dist in [
            Distribution::Gaussian,
            Distribution::Logistic,
            Distribution::Cauchy,
        ] {
            for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let x = quantile(dist, p);
                let p_back = cdf(dist, x);
                assert!(
                    (p - p_back).abs() < 1e-4,
                    "CDF(quantile({p})) = {p_back} for {dist:?}"
                );
            }
        }
    }

    // ── Lloyd-Max tests ────────────────────────────────────────────────────

    #[test]
    fn lloyd_max_level_counts() {
        for n in [1, 2, 3, 4, 8, 16] {
            let cb = lloyd_max_codebook(LloydMaxDist::Gaussian, n).unwrap();
            assert_eq!(cb.representatives.len(), n);
            assert_eq!(cb.boundaries.len(), n.saturating_sub(1));
        }
    }

    #[test]
    fn lloyd_max_zero_levels_is_error() {
        assert!(lloyd_max_codebook(LloydMaxDist::Gaussian, 0).is_err());
    }

    #[test]
    fn lloyd_max_boundaries_sorted() {
        for dist in [
            LloydMaxDist::Gaussian,
            LloydMaxDist::Logistic,
            LloydMaxDist::Cauchy,
        ] {
            let cb = lloyd_max_codebook(dist, 8).unwrap();
            for w in cb.boundaries.windows(2) {
                assert!(w[0] < w[1], "unsorted boundaries for {dist:?}: {w:?}");
            }
        }
    }

    #[test]
    fn lloyd_max_representatives_within_boundaries() {
        for dist in [
            LloydMaxDist::Gaussian,
            LloydMaxDist::Logistic,
            LloydMaxDist::Cauchy,
        ] {
            let cb = lloyd_max_codebook(dist, 8).unwrap();
            let n = cb.representatives.len();
            for i in 0..n {
                let lo = if i == 0 {
                    f64::NEG_INFINITY
                } else {
                    cb.boundaries[i - 1]
                };
                let hi = if i == n - 1 {
                    f64::INFINITY
                } else {
                    cb.boundaries[i]
                };
                let r = cb.representatives[i];
                assert!(
                    r > lo && r < hi,
                    "rep {r} not in ({lo}, {hi}) for {dist:?} level {i}"
                );
            }
        }
    }

    #[test]
    fn lloyd_max_gaussian_two_level() {
        let cb = lloyd_max_codebook(LloydMaxDist::Gaussian, 2).unwrap();
        assert_eq!(cb.boundaries.len(), 1);
        assert!(cb.boundaries[0].abs() < 1e-6);
        let expected = (2.0 / PI).sqrt();
        assert!((cb.representatives[0] + expected).abs() < 1e-5);
        assert!((cb.representatives[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn lloyd_max_gaussian_four_level() {
        let cb = lloyd_max_codebook(LloydMaxDist::Gaussian, 4).unwrap();
        assert!(
            cb.boundaries[1].abs() < 1e-6,
            "middle boundary should be ~0"
        );
        assert!((cb.boundaries[2] - 0.9816).abs() < 0.01);
        assert!((cb.representatives[3] - 1.510).abs() < 0.01);
        assert!((cb.representatives[2] - 0.4528).abs() < 0.01);
    }

    #[test]
    fn lloyd_max_quantize_dequantize_roundtrip() {
        let cb = lloyd_max_codebook(LloydMaxDist::Gaussian, 8).unwrap();
        for (i, &r) in cb.representatives.iter().enumerate() {
            let idx = lloyd_max_quantize(r, &cb.boundaries);
            assert_eq!(idx, i);
            let recovered = lloyd_max_dequantize(idx, &cb.representatives);
            assert!((recovered - r).abs() < 1e-14);
        }
    }

    #[test]
    fn lloyd_max_symmetric_codebook() {
        for dist in [
            LloydMaxDist::Gaussian,
            LloydMaxDist::Logistic,
            LloydMaxDist::Cauchy,
        ] {
            let cb = lloyd_max_codebook(dist, 8).unwrap();
            let n = cb.representatives.len();
            for i in 0..n / 2 {
                let lo = cb.representatives[i];
                let hi = cb.representatives[n - 1 - i];
                assert!(
                    (lo + hi).abs() < 1e-6,
                    "{dist:?}: reps not symmetric: {lo} vs {hi}"
                );
            }
        }
    }

    #[test]
    fn lloyd_max_cauchy_wider_than_gaussian() {
        let gauss = lloyd_max_codebook(LloydMaxDist::Gaussian, 8).unwrap();
        let cauchy = lloyd_max_codebook(LloydMaxDist::Cauchy, 8).unwrap();
        let g_outer = gauss.representatives.last().unwrap().abs();
        let c_outer = cauchy.representatives.last().unwrap().abs();
        assert!(c_outer > g_outer);
    }

    #[test]
    fn lloyd_max_convergence() {
        for dist in [
            LloydMaxDist::Gaussian,
            LloydMaxDist::Logistic,
            LloydMaxDist::Cauchy,
        ] {
            let cb = lloyd_max_codebook(dist, 16).unwrap();
            assert!(
                cb.iterations < 1000,
                "{dist:?}: took {} iterations (did not converge)",
                cb.iterations
            );
        }
    }

    #[test]
    fn erfc_approx_accuracy() {
        let cases = [
            (0.0, 1.0),
            (1.0, 0.157299207050285),
            (2.0, 0.004677734981047),
            (-1.0, 1.842700792949715),
        ];
        for (x, expected) in cases {
            let got = erfc_approx(x);
            assert!(
                (got - expected).abs() < 2e-7,
                "erfc({x}): expected {expected}, got {got}"
            );
        }
    }
}
