//! Matryoshka scalar quantization (Nair et al. 2025, arXiv:2502.06786).
//!
//! A value is quantized once to a `max_bits`-bit (c-bit) affine integer code. A
//! lower-precision r-bit view (`r < c`) is the r most-significant bits of that
//! code, via the slice operation (paper Eq. 6):
//!
//! ```text
//! S(q, r) = clamp(round(q / 2^(c-r)), 0, 2^r - 1)
//! ```
//!
//! Note `round` (not floor) and the `clamp`, which the paper requires because
//! rounding can overflow the r-bit range. All precisions share ONE scale and
//! zero-point (the c-bit affine parameters); the r-bit reconstruction reuses the
//! c-bit grid by left-shifting the slice back.
//!
//! ## What MatQuant actually adds
//!
//! Plain min/max c-bit quantization is mechanically sliceable, but its low-bit
//! slices reconstruct POORLY (the paper's "sliced" baseline). MatQuant's
//! contribution is to choose the quantization range so that a *weighted*
//! reconstruction error across precisions is jointly minimized.
//! `MatryoshkaQuantizer::fit` does this with a clip-range search: the scalar
//! analog of the paper's gradient joint-optimization over model weights. This is
//! NOT the full model-training method, and the int2-specific tricks from the
//! paper (co-distillation, single-precision MatQuant, an extra outlier bit) are
//! out of scope here.
//!
//! ## Contract
//!
//! - Codes are `u8`; `max_bits` is in `1..=8`.
//! - `slice` extracts MSBs with rounding (paper Eq. 6), so prefix-nesting is
//!   approximate, not exact bit-truncation.
//! - All precisions share the affine scale/zero-point; reconstruction is the
//!   c-bit grid point of the sliced code.

use crate::{Result, VQuantError};

/// A scalar quantizer whose c-bit codes slice to lower precisions by MSB.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatryoshkaQuantizer {
    lo: f32,
    hi: f32,
    max_bits: u8,
}

impl MatryoshkaQuantizer {
    /// Build a quantizer over the affine range `[lo, hi]` with `max_bits`-bit
    /// codes (`max_bits` in `1..=8`).
    pub fn new(lo: f32, hi: f32, max_bits: u8) -> Result<Self> {
        if !(1..=8).contains(&max_bits) {
            return Err(VQuantError::InvalidConfig {
                field: "max_bits",
                reason: "must be in 1..=8",
            });
        }
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return Err(VQuantError::InvalidConfig {
                field: "range",
                reason: "require finite lo < hi",
            });
        }
        Ok(Self { lo, hi, max_bits })
    }

    /// Naive min/max range over `values` (the paper's weak "sliced" baseline).
    pub fn fit_minmax(values: &[f32], max_bits: u8) -> Result<Self> {
        let (lo, hi) = finite_min_max(values)?;
        Self::new(lo, hi, max_bits)
    }

    /// Fit the quantization range to jointly minimize a weighted reconstruction
    /// error across the given `precisions`, the scalar analog of MatQuant's
    /// joint optimization. Searches symmetric clip quantiles and keeps the range
    /// with the lowest weighted error.
    ///
    /// `precisions` are bit-widths (each `<= max_bits`); `weights` are their
    /// relative loss weights (the paper's lambda_r). The default emphasis on the
    /// lowest precision matches the paper's finding that a higher weight on the
    /// smallest bit-width is essential.
    pub fn fit(values: &[f32], max_bits: u8, precisions: &[u8], weights: &[f32]) -> Result<Self> {
        if precisions.is_empty() || precisions.len() != weights.len() {
            return Err(VQuantError::InvalidConfig {
                field: "precisions/weights",
                reason: "non-empty and equal length required",
            });
        }
        for &r in precisions {
            if !(1..=max_bits).contains(&r) {
                return Err(VQuantError::InvalidConfig {
                    field: "precisions",
                    reason: "each precision must be in 1..=max_bits",
                });
            }
        }
        let (lo0, hi0) = finite_min_max(values)?;
        let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Candidate symmetric clip quantiles. gamma = 0 is the min/max baseline.
        const GAMMAS: [f32; 7] = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1];
        let mut best: Option<(f32, Self)> = None;
        for &gamma in &GAMMAS {
            let (lo, hi) = if gamma == 0.0 {
                (lo0, hi0)
            } else {
                (quantile(&sorted, gamma), quantile(&sorted, 1.0 - gamma))
            };
            let Ok(q) = Self::new(lo, hi, max_bits) else {
                continue;
            };
            let err = q.weighted_mse(values, precisions, weights);
            if best.as_ref().map_or(true, |(b, _)| err < *b) {
                best = Some((err, q));
            }
        }
        best.map(|(_, q)| q).ok_or(VQuantError::InvalidConfig {
            field: "values",
            reason: "no valid clip range found",
        })
    }

    /// Maximum code bit-width.
    pub fn max_bits(&self) -> u8 {
        self.max_bits
    }

    /// Quantize a value to a full-precision `max_bits` affine code.
    pub fn quantize(&self, value: f32) -> u8 {
        let levels = ((1u16 << self.max_bits) - 1) as f32; // 2^c - 1
        let t = ((value - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0);
        (t * levels).round() as u8
    }

    /// Slice a full c-bit code to its `bits` most-significant bits (Eq. 6).
    pub fn slice(&self, code: u8, bits: u8) -> u8 {
        let r = bits.clamp(1, self.max_bits);
        let shift = self.max_bits - r;
        let max_r = (1u16 << r) - 1;
        let sliced = ((code as f32) / ((1u16 << shift) as f32)).round() as u16;
        sliced.min(max_r) as u8
    }

    /// Reconstruct a value from an r-bit sliced code, on the shared c-bit grid.
    pub fn dequantize(&self, code_r: u8, bits: u8) -> f32 {
        let r = bits.clamp(1, self.max_bits);
        let shift = self.max_bits - r;
        let levels = ((1u16 << self.max_bits) - 1) as f32; // 2^c - 1
        let grid_value = (code_r as u16 * (1u16 << shift)) as f32; // left-shift back
        self.lo + (grid_value / levels) * (self.hi - self.lo)
    }

    /// Quantize a value and reconstruct it at `bits` precision (quantize, slice,
    /// dequantize) from the single stored code.
    pub fn reconstruct(&self, value: f32, bits: u8) -> f32 {
        let code = self.quantize(value);
        let sliced = self.slice(code, bits);
        self.dequantize(sliced, bits)
    }

    /// Weighted mean squared reconstruction error across `precisions`.
    pub fn weighted_mse(&self, values: &[f32], precisions: &[u8], weights: &[f32]) -> f32 {
        let mut total = 0.0f32;
        let mut wsum = 0.0f32;
        for (&r, &w) in precisions.iter().zip(weights.iter()) {
            let mut se = 0.0f32;
            let mut n = 0usize;
            for &v in values.iter().filter(|v| v.is_finite()) {
                let d = v - self.reconstruct(v, r);
                se += d * d;
                n += 1;
            }
            if n > 0 {
                total += w * (se / n as f32);
                wsum += w;
            }
        }
        if wsum > 0.0 {
            total / wsum
        } else {
            0.0
        }
    }
}

fn finite_min_max(values: &[f32]) -> Result<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return Err(VQuantError::InvalidConfig {
            field: "values",
            reason: "no finite values",
        });
    }
    if hi <= lo {
        hi = lo + 1.0; // degenerate constant input: widen to a unit range
    }
    Ok((lo, hi))
}

/// Linear-interpolated quantile of a pre-sorted slice (`q` in `[0, 1]`).
fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_matches_eq6() {
        // c = 8: code 200, slice to 2 bits = clamp(round(200/64), 0, 3) = 3.
        let q = MatryoshkaQuantizer::new(0.0, 1.0, 8).unwrap();
        assert_eq!(q.slice(200, 2), 3);
        // code 95, slice to 4 bits = clamp(round(95/16), 0, 15) = 6.
        assert_eq!(q.slice(95, 4), 6);
        // full precision is identity.
        assert_eq!(q.slice(123, 8), 123);
    }

    #[test]
    fn reconstruction_error_decreases_with_bits() {
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let q = MatryoshkaQuantizer::fit_minmax(&values, 8).unwrap();
        let e8 = q.weighted_mse(&values, &[8], &[1.0]);
        let e4 = q.weighted_mse(&values, &[4], &[1.0]);
        let e2 = q.weighted_mse(&values, &[2], &[1.0]);
        assert!(e8 <= e4, "8-bit error {e8} should be <= 4-bit {e4}");
        assert!(e4 <= e2, "4-bit error {e4} should be <= 2-bit {e2}");
    }

    #[test]
    fn fit_beats_minmax_at_low_bits() {
        // Heavy-tailed input: a dense Gaussian-ish core plus a few far outliers.
        // Min/max stretches the range to cover outliers, starving the low-bit
        // grid; clipping (fit) should reconstruct the dense core better at int2.
        let mut values: Vec<f32> = Vec::new();
        for i in 0..200 {
            // deterministic pseudo-spread in [-1, 1]
            let x = ((i as f32) * 0.0173).sin();
            values.push(x);
        }
        values.extend_from_slice(&[40.0, -38.0, 45.0]); // outliers

        let precisions = [8u8, 4, 2];
        let weights = [0.1f32, 0.1, 1.0]; // emphasize int2, per the paper
        let fitted = MatryoshkaQuantizer::fit(&values, 8, &precisions, &weights).unwrap();
        let baseline = MatryoshkaQuantizer::fit_minmax(&values, 8).unwrap();

        // Compare int2 error on the dense core only (what we actually care about).
        let core: Vec<f32> = values.iter().copied().filter(|v| v.abs() <= 1.0).collect();
        let e_fit = fitted.weighted_mse(&core, &[2], &[1.0]);
        let e_base = baseline.weighted_mse(&core, &[2], &[1.0]);
        assert!(
            e_fit < e_base,
            "fit int2 core error {e_fit} should beat min/max {e_base}"
        );
    }

    #[test]
    fn rejects_bad_config() {
        assert!(MatryoshkaQuantizer::new(0.0, 1.0, 0).is_err());
        assert!(MatryoshkaQuantizer::new(0.0, 1.0, 9).is_err());
        assert!(MatryoshkaQuantizer::new(1.0, 1.0, 8).is_err());
        assert!(MatryoshkaQuantizer::fit_minmax(&[], 8).is_err());
    }

    use proptest::prelude::*;

    proptest! {
        // The robustly-true MSB-slice invariants (not the rounding-approximate
        // prefix-nesting the contract explicitly disclaims): the sliced code
        // stays in range, full precision is the identity, and slicing is order
        // preserving in the code. These pin Eq. 6 across the whole code space,
        // where slice_matches_eq6 only checks three fixtures.

        /// `slice(code, b)` always lands in `[0, 2^b - 1]`.
        #[test]
        fn slice_stays_in_b_bit_range(code in any::<u8>(), bits in 1u8..=8) {
            let q = MatryoshkaQuantizer::new(0.0, 1.0, 8).unwrap();
            let sliced = q.slice(code, bits);
            let max_code = (1u16 << bits) - 1;
            prop_assert!(
                u16::from(sliced) <= max_code,
                "slice({code}, {bits}) = {sliced} exceeds {max_code}"
            );
        }

        /// Slicing to the full precision returns the code unchanged.
        #[test]
        fn slice_at_full_precision_is_identity(code in any::<u8>()) {
            let q = MatryoshkaQuantizer::new(0.0, 1.0, 8).unwrap();
            prop_assert_eq!(q.slice(code, 8), code);
        }

        /// Slicing is monotone non-decreasing in the code: a larger code never
        /// slices to a smaller value at the same precision (MSB extraction
        /// preserves order, so higher codes stay higher after truncation).
        #[test]
        fn slice_preserves_code_order(a in any::<u8>(), b in any::<u8>(), bits in 1u8..=8) {
            let q = MatryoshkaQuantizer::new(0.0, 1.0, 8).unwrap();
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            prop_assert!(
                q.slice(lo, bits) <= q.slice(hi, bits),
                "slice not monotone at {bits} bits: slice({lo})={}, slice({hi})={}",
                q.slice(lo, bits),
                q.slice(hi, bits)
            );
        }
    }
}
