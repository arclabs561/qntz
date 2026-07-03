# Changelog

## [Unreleased]

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
