//! Property tests for distributional quantization.

#![cfg(feature = "distquant")]

use proptest::prelude::*;
use qntz::distquant::{self, Distribution};

proptest! {
    /// Codebook should always be monotonically non-decreasing.
    #[test]
    fn prop_codebook_monotonic(
        n in 2usize..=64,
        beta in 0.25f64..=5.0,
        dist_idx in 0usize..4,
    ) {
        let dists = [
            Distribution::Gaussian,
            Distribution::Logistic,
            Distribution::Gumbel,
            Distribution::Cauchy,
        ];
        let dist = dists[dist_idx];
        let cb = distquant::build_codebook(n, dist, Some(beta)).unwrap();
        for i in 1..cb.len() {
            prop_assert!(cb[i] >= cb[i - 1], "Codebook not monotonic at {i}: {:?}", &cb[..=i]);
        }
    }

    /// Quantize-dequantize roundtrip should have bounded error.
    #[test]
    fn prop_roundtrip_bounded_error(
        bits in 2u8..=6,
        len in 1usize..=100,
        seed in 0u64..1000,
    ) {
        // Generate deterministic pseudo-random data
        let data: Vec<f64> = (0..len)
            .map(|i| ((seed.wrapping_mul(17).wrapping_add(i as u64)) as f64 / 100.0) - 5.0)
            .collect();

        let (codes, info) = distquant::quantize(&data, bits, Distribution::Gaussian, None);
        let mse = distquant::quantization_mse(&data, &codes, &info);

        // MSE should be finite and non-negative
        prop_assert!(mse.is_finite(), "MSE is not finite: {mse}");
        prop_assert!(mse >= 0.0, "MSE is negative: {mse}");
    }

    /// Pack-unpack roundtrip should be lossless.
    #[test]
    fn prop_pack_unpack_roundtrip(
        bits in 1u8..=8,
        len in 1usize..=50,
    ) {
        let max_val = (1u32 << bits) - 1;
        let codes: Vec<u32> = (0..len).map(|i| (i as u32) % (max_val + 1)).collect();
        let packed = distquant::pack_codes(&codes, bits);
        let unpacked = distquant::unpack_codes(&packed, bits, codes.len());
        prop_assert_eq!(&codes, &unpacked);
    }

    /// More bits should give equal or lower MSE.
    #[test]
    fn prop_more_bits_lower_mse(
        len in 10usize..=50,
        seed in 0u64..100,
    ) {
        let data: Vec<f64> = (0..len)
            .map(|i| ((seed.wrapping_mul(31).wrapping_add(i as u64)) as f64 / 50.0) - 5.0)
            .collect();

        let (c2, i2) = distquant::quantize(&data, 2, Distribution::Gaussian, None);
        let (c4, i4) = distquant::quantize(&data, 4, Distribution::Gaussian, None);
        let mse2 = distquant::quantization_mse(&data, &c2, &i2);
        let mse4 = distquant::quantization_mse(&data, &c4, &i4);

        prop_assert!(mse4 <= mse2 + 1e-10, "4-bit ({mse4}) should beat 2-bit ({mse2})");
    }
}
