//! Regressions for ternary target sparsity and the distquant Exponential prior.
//!
//! Each regression runs whenever its quantizer is available.

#[cfg(any(feature = "ternary", feature = "distquant"))]
#[path = "bughunt_ternary_distquant/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "ternary", feature = "distquant")))]
#[test]
fn bughunt_suite_is_not_silently_empty() {
    // Always-true invariant: the padding check that guards the ternary code
    // path rejects an undersized destination buffer.
    let codes = [1u8; 9];
    let mut too_small = vec![0u8; 1];
    assert!(
        qntz::simd_ops::pack_binary_fast(&codes, &mut too_small).is_err(),
        "packing 9 codes into 1 byte must fail"
    );

    eprintln!(
        "tests/bughunt_ternary_distquant.rs: enable ternary and distquant to run both quantizer regressions"
    );
}
