//! Property tests for low-bit code packing.
//!
//! Packing and portable distance properties run in the default build.
//! Quantizer-specific properties run when their feature is selected.

#[path = "property_tests/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "rabitq", feature = "ternary")))]
#[test]
fn property_tests_are_not_silently_empty() {
    // Always-true invariant: bit packing roundtrips for any width, with no
    // optimization features enabled.
    let codes: Vec<u8> = (0..11u8).map(|i| i % 2).collect();
    let mut packed = vec![0u8; codes.len().div_ceil(8)];
    qntz::simd_ops::pack_binary_fast(&codes, &mut packed).expect("pack 11 codes");
    let mut unpacked = vec![0u8; codes.len()];
    qntz::simd_ops::unpack_binary_fast(&packed, &mut unpacked, codes.len()).expect("unpack");
    assert_eq!(unpacked, codes, "binary pack/unpack must roundtrip");

    eprintln!("tests/property_tests.rs: enable rabitq and ternary to run all quantizer properties");
}
