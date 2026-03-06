use proptest::prelude::*;

use qntz::simd_ops;

proptest! {
    #[test]
    fn prop_pack_unpack_binary_roundtrip(codes in prop::collection::vec(prop_oneof![Just(0u8), Just(1u8)], 0..256)) {
        let packed_len = codes.len().div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        simd_ops::pack_binary_fast(&codes, &mut packed);

        let mut unpacked = vec![0u8; codes.len()];
        simd_ops::unpack_binary_fast(&packed, &mut unpacked, codes.len());

        prop_assert_eq!(unpacked, codes);
    }

    #[test]
    fn prop_hamming_distance_symmetric(a in prop::collection::vec(any::<u8>(), 0..128),
                                      b in prop::collection::vec(any::<u8>(), 0..128)) {
        let dab = simd_ops::hamming_distance(&a, &b);
        let dba = simd_ops::hamming_distance(&b, &a);
        prop_assert_eq!(dab, dba);
    }

    #[test]
    fn prop_extended_pack_roundtrip(
        dim in 1usize..128,
        ex_bits in 1usize..8,
    ) {
        let max_val = (1u16 << ex_bits) - 1;
        let codes: Vec<u16> = (0..dim).map(|i| (i as u16) % (max_val + 1)).collect();
        let packed_len = (dim * ex_bits).div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        simd_ops::pack_extended_interleaved(&codes, &mut packed, ex_bits);
        let mut unpacked = vec![0u16; dim];
        simd_ops::unpack_extended_interleaved(&packed, &mut unpacked, dim, ex_bits);
        prop_assert_eq!(&codes, &unpacked);
    }

    #[test]
    fn prop_hamming_triangle_inequality(
        a in prop::collection::vec(any::<u8>(), 1..64),
    ) {
        let b: Vec<u8> = a.iter().map(|&x| x.wrapping_add(1)).collect();
        let c: Vec<u8> = a.iter().map(|&x| x.wrapping_add(2)).collect();
        let d_ab = simd_ops::hamming_distance(&a, &b);
        let d_bc = simd_ops::hamming_distance(&b, &c);
        let d_ac = simd_ops::hamming_distance(&a, &c);
        prop_assert!(d_ac <= d_ab + d_bc,
            "triangle inequality violated: d(a,c)={} > d(a,b)={} + d(b,c)={}", d_ac, d_ab, d_bc);
    }

    #[test]
    fn prop_asymmetric_l2_nonneg(
        query in prop::collection::vec(-100.0f32..100.0, 8..128),
    ) {
        let dim = query.len();
        let packed_len = dim.div_ceil(8);
        let codes = vec![0xFFu8; packed_len];
        let l2 = simd_ops::asymmetric_l2_squared(&query[..dim], &codes);
        prop_assert!(l2 >= 0.0, "L2 squared distance is negative: {}", l2);
    }
}

#[cfg(feature = "ternary")]
mod ternary_props {
    use super::*;
    use qntz::ternary::{TernaryConfig, TernaryQuantizer};

    proptest! {
        #[test]
        fn prop_ternary_values_in_domain(
            dim in 1usize..128,
            xs in prop::collection::vec(prop::num::f32::NORMAL, 1..128),
        ) {
            let mut v = xs;
            v.truncate(dim);
            if v.len() < dim {
                v.resize(dim, 0.0);
            }

            // Keep it simple/robust: no normalization, fixed thresholds.
            let config = TernaryConfig { normalize: false, ..TernaryConfig::default() };
            let quantizer = TernaryQuantizer::new(dim, config);
            let q = quantizer.quantize(&v).unwrap();

            prop_assert_eq!(q.dimension(), dim);
            prop_assert!(q.memory_bytes() <= dim.div_ceil(4));

            for i in 0..dim {
                let val = q.get(i);
                prop_assert!(val == -1 || val == 0 || val == 1);
            }

            let s = q.sparsity();
            prop_assert!((0.0..=1.0).contains(&s));
        }
    }
}

#[cfg(feature = "rabitq")]
mod rabitq_props {
    use super::*;
    use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

    proptest! {
        #[test]
        fn prop_rabitq_lengths_and_code_ranges(
            dim in 1usize..64,
            total_bits in prop_oneof![Just(1usize), Just(4usize), Just(8usize)],
            xs in prop::collection::vec(prop::num::f32::NORMAL, 1..64),
        ) {
            let mut v = xs;
            v.truncate(dim);
            if v.len() < dim {
                v.resize(dim, 0.0);
            }

            // Keep values finite-ish to avoid spurious NaNs propagating through the math.
            for x in &mut v {
                *x = x.clamp(-10.0, 10.0);
            }

            let config = match total_bits {
                1 => RaBitQConfig::binary(),
                4 => RaBitQConfig::bits4(),
                8 => RaBitQConfig::bits8(),
                _ => unreachable!(),
            };

            let quantizer = RaBitQQuantizer::with_config(dim, 42, config).unwrap();
            let q = quantizer.quantize(&v).unwrap();

            let ex_bits = total_bits.saturating_sub(1);
            let expected_binary_len = dim.div_ceil(8);
            let expected_extended_len = if ex_bits == 0 { 0 } else { (dim * ex_bits).div_ceil(8) };

            prop_assert_eq!(q.binary_codes.len(), expected_binary_len);
            prop_assert_eq!(q.extended_codes.len(), expected_extended_len);
            prop_assert_eq!(q.codes.len(), dim);

            let max_code = (1u16 << total_bits) - 1;
            for &c in &q.codes {
                prop_assert!(c <= max_code);
            }

            let d = quantizer.approximate_distance(&v, &q).unwrap();
            prop_assert!(d.is_finite());
            prop_assert!(d >= 0.0);
        }
    }
}
