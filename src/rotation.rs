//! Shared orthogonal rotation matrix generation.
//!
//! Used by `rabitq` and `binary` to produce random orthonormal bases via
//! Gram-Schmidt on Gaussian random vectors.

/// Generate a `nrows × dim` row-major orthonormal matrix.
///
/// When `nrows == dim`, the result is a full square orthogonal matrix.
/// When `nrows < dim`, only the first `nrows` orthonormal vectors are returned
/// (useful for dimensionality-reducing projections).
///
/// The `f64::EPSILON` guard on the norm prevents degenerate near-zero vectors
/// from being scaled to garbage; they fall back to the standard basis vector at
/// index `row % dim`.
pub(crate) fn orthogonal_rotation_matrix(dim: usize, nrows: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut state = seed;
    let mut next_gaussian = || -> f32 {
        // Box-Muller using a hash-chain as the RNG.
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        let u1 = (state as f64) / (u64::MAX as f64);
        let mut hasher2 = DefaultHasher::new();
        state.hash(&mut hasher2);
        state = hasher2.finish();
        let u2 = (state as f64) / (u64::MAX as f64);
        ((-2.0 * u1.max(f64::EPSILON).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };

    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(nrows);

    for i in 0..nrows {
        let mut v: Vec<f32> = (0..dim).map(|_| next_gaussian()).collect();

        // Gram-Schmidt: subtract projections onto all previous basis vectors.
        for b in &basis {
            let dot: f32 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
            basis.push(v);
        } else {
            // Degenerate: fall back to standard basis vector.
            let mut e = vec![0.0f32; dim];
            e[i % dim] = 1.0;
            basis.push(e);
        }
    }

    // Flatten into row-major matrix.
    let mut rotation = vec![0.0f32; nrows * dim];
    for (row_idx, row) in basis.iter().enumerate() {
        let offset = row_idx * dim;
        rotation[offset..offset + dim].copy_from_slice(row);
    }
    rotation
}
