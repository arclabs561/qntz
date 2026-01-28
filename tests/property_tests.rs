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
            prop_assert!(q.memory_bytes() <= (dim + 3) / 4);

            for i in 0..dim {
                let val = q.get(i);
                prop_assert!(val == -1 || val == 0 || val == 1);
            }

            let s = q.sparsity();
            prop_assert!(s >= 0.0 && s <= 1.0);
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
            let expected_binary_len = (dim + 7) / 8;
            let expected_extended_len = if ex_bits == 0 { 0 } else { (dim * ex_bits + 7) / 8 };

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
