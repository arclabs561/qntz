# Changelog

## [Unreleased]

## [0.2.2] - Release candidate

This section describes the prepared release; publication is pending.

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
