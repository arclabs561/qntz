# Changelog

## [Unreleased]

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
