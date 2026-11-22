# Changelog

## [Unreleased]

## [0.3.1] - 2026-11-04

### Changed
- Binary asymmetric scoring (`asymmetric_inner_product`,
  `BinaryQuantizer::asymmetric_distance_prerotated`, the batch L2) uses
  innr's `sign_dot` with the `simd` feature and an eight-lane loop without
  it, instead of one accumulator with a branch per bit. Per code, innr's
  probe measured 4.5x (128 bits) to 6.5x (1536 bits) on an Apple M3 and 5x
  to 15.8x on an i7-8559U; `batch_asymmetric_l2_into` over 256 codes of 768
  bits went from 1.62 ms to 95 us on the M3 (separate runs, machine loaded).
- `batch_asymmetric_l2_into` computes the query norm once per batch, not
  once per code.
- The optional `innr` dependency is 0.7.7 (was 0.6.3), so builds with
  `vicinity` no longer carry two innr versions once both move.

## [0.3.0] - 2026-11-02

### Added
- RaBitQ codes serialise: `QuantizedVector` and `EdgeQuantizedVector` have
  `to_bytes` / `from_bytes`, validating magic, version, bit width, lengths,
  finite correction factors and extended-code ranges.
- `BinaryQuantizer::rotate_query` and `asymmetric_distance_prerotated`, so a
  scan rotates the query once instead of once per code;
  `asymmetric_distance` is their composition.

### Changed
- `VQuantError` gains `MalformedCode` and is `#[non_exhaustive]`.

## [0.2.3] - 2026-10-30

### Fixed
- `TernaryQuantizer::fit` with `target_sparsity` fits thresholds on the same
  centered and L2-normalized values that `quantize` thresholds. With the
  default `normalize: true`, a 0.5 target previously produced sparsity near 1.
- `distquant::quantize` with `Distribution::Exponential` centers the codebook
  on standardized data. Exp(1) quantiles are all non-negative, so values
  below the mean all mapped to code 0 (67% of an Exp(1) sample; MSE 0.351,
  now 0.055). `build_codebook` still returns raw quantiles.
## [0.3.0] - 2026-11-02

### Added
- `QuantizedVector::to_bytes` / `from_bytes` / `encode_into` / `encoded_len` and
  `EdgeQuantizedVector::to_bytes` / `from_bytes`, so a built code can be
  persisted and reloaded instead of being re-derived from the vectors it came
  from. The derived `codes` vector is recomputed on load rather than stored, and
  every checkable property is checked: magic, version, bit width, lengths against
  the declared dimension, finiteness of the correction factors.
- `BinaryQuantizer::rotate_query` and `asymmetric_distance_prerotated`, which
  split the rotate-then-score path so a caller can rotate once per query instead
  of once per candidate. `asymmetric_distance` is now their composition and
  behaves identically.
- `VQuantError::MalformedCode`, carrying a static reason.

### Changed
- `VQuantError` is now `#[non_exhaustive]`. Decode failures are expected to grow
  with the number of formats that can be serialised, and callers should have a
  fallback arm. This is the reason for the minor version bump.

## [0.2.2] - 2026-10-22

### Added
- Checked alternatives for distributional quantization and code packing,
  extended-bit packing/unpacking and inner products, ternary construction,
  RaBitQ constant scaling, and pre-rotated RaBitQ scoring.
- `RaBitQQuantizer::squared_distance_error_margin` scales the stored error
  coefficient by the query residual norm. The coefficient alone is not an
  error radius for arbitrary queries.
- A `rabitq_statistical_eval` example and tests for distance-estimator bias
  and error-margin coverage.

### Changed
- Stabilized dense rotation generation with an explicitly specified random
  generator and reorthogonalization. **The same seed produces different
  rotations than in 0.2.1.** Seeded construction is repeatable within a crate
  version, not a cross-version serialization format. Persist trained
  quantizers when exact matrix reuse is required; do not combine previously
  generated codes with a newly generated rotation.
- RaBitQ and ternary operations validate dimensions, finite inputs and
  configuration before computation; invalid inputs can now return errors
  rather than propagate nonfinite values. Packing and scoring APIs document
  their compatibility-wrapper behavior and provide checked alternatives.
- Qualified distributional codebooks as quantile-based rather than optimal,
  and removed workload-independent recall claims from RaBitQ bit presets.
- Updated the `entropy_coded_quantization` example to use `ans 0.4`.

### Fixed
- `RaBitQQuantizer::edge_distance_term_prerotated` no longer clamps the edge
  term at zero. The term estimates `||q-v||^2 - ||q-u||^2` and is negative
  whenever a child is nearer the query than its parent; the clamp tied all
  such children at `||q-u||^2` and lost their ordering.
- Correct the distance-quality test oracle to average tied ranks; retain its
  zero score for constant predictions. This is a test-only change to the
  evaluation helper, not a production quantizer change.
  ([93eb438](https://github.com/arclabs561/qntz/commit/93eb438edf56f79987f13a3433dce24f5d5aaaad))
- Restored the full Apache-2.0 license text and added a dual-license pointer.

## [0.2.1] - 2026-07-03

### Added
- `simd_ops::batch_hamming_distances_into`,
  `simd_ops::batch_asymmetric_l2_into`, and
  `adaptive::PackedBatch::asymmetric_distances_into` let repeated scan loops
  reuse caller-provided output buffers instead of allocating a new distance
  vector per query.
- `adaptive::PackedBatch::scan_plan` and `PackedBatchScanPlan` cache per-vector
  code statistics for repeated adaptive scans over the same packed batch.

### Changed
- Bumped the optional `innr` dependency (behind the `simd` feature) to 0.6.
  Recorded retroactively: the bump shipped without a changelog entry (the
  last recorded innr version was 0.4, in 0.1.9).
- Changed adaptive packed scan plans to use `innr`'s mixed f32/u8 dot primitive
  when the `simd` feature is enabled.

## [0.2.0] - 2026-06-24

### Added
- `matryoshka` feature: `MatryoshkaQuantizer`, scalar Matryoshka quantization
  (Nair et al. 2025, arXiv:2502.06786). One c-bit code is sliceable to lower
  precisions via MSB slicing (paper Eq. 6), and `fit` searches a clip range that
  minimizes a weighted reconstruction error across precisions (the scalar analog
  of the paper's joint optimization, which beats naive min/max at low bits).
- Added examples for RaBitQ bit-width error budget and adaptive scan-distance
  error.

## [0.1.9] - 2026-06-10

### Changed
- Bumped `innr` to 0.4.
- README and CONTRIBUTING polish; clippy/rustfmt added to CI; publish gated on cargo-semver-checks.

Earlier releases predate this changelog; see git history.
