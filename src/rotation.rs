//! Shared orthogonal rotation matrix generation.
//!
//! Used by `rabitq` and `binary` to produce random orthonormal bases from
//! Gaussian random vectors. Persist trained quantizers when exact rotation
//! reuse matters: seeded construction is deterministic within a crate version,
//! but the generated matrix is not a cross-version serialization format.

/// Small, explicitly specified generator used for reproducible matrix creation.
///
/// SplitMix64 is suitable here because rotation generation needs stable,
/// well-distributed samples rather than cryptographic unpredictability.
struct StableRng {
    state: u64,
    spare_gaussian: Option<f64>,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_gaussian: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample strictly inside `(0, 1)` using the high 53 random bits.
    fn open_unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        (((self.next_u64() >> 11) as f64) + 0.5) * SCALE
    }

    fn gaussian(&mut self) -> f64 {
        if let Some(spare) = self.spare_gaussian.take() {
            return spare;
        }

        let radius = (-2.0 * self.open_unit().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.open_unit();
        let (sin, cos) = angle.sin_cos();
        self.spare_gaussian = Some(radius * sin);
        radius * cos
    }
}

fn subtract_projections(vector: &mut [f64], basis: &[Vec<f64>]) {
    for row in basis {
        let projection: f64 = vector.iter().zip(row).map(|(a, b)| a * b).sum();
        for (value, component) in vector.iter_mut().zip(row) {
            *value -= projection * component;
        }
    }
}

/// Reorthogonalize twice to control the loss of orthogonality from finite
/// precision modified Gram-Schmidt.
fn reorthogonalize(vector: &mut [f64], basis: &[Vec<f64>]) {
    subtract_projections(vector, basis);
    subtract_projections(vector, basis);
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Choose the coordinate vector with the largest residual after projection.
/// This remains orthogonal to the existing basis even if a random candidate is
/// numerically degenerate.
fn orthogonal_fallback(dim: usize, basis: &[Vec<f64>]) -> Vec<f64> {
    let mut best = Vec::new();
    let mut best_norm = -1.0f64;

    for index in 0..dim {
        let mut candidate = vec![0.0; dim];
        candidate[index] = 1.0;
        reorthogonalize(&mut candidate, basis);
        let candidate_norm = norm(&candidate);
        if candidate_norm > best_norm {
            best = candidate;
            best_norm = candidate_norm;
        }
    }

    assert!(
        best_norm > f64::EPSILON,
        "cannot extend an orthonormal basis beyond its dimension"
    );
    for value in &mut best {
        *value /= best_norm;
    }
    best
}

/// Generate a `nrows × dim` row-major orthonormal matrix.
///
/// When `nrows == dim`, the result is a full square orthogonal matrix.
/// When `nrows < dim`, only the first `nrows` orthonormal vectors are returned
/// (useful for dimensionality-reducing projections).
pub(crate) fn orthogonal_rotation_matrix(dim: usize, nrows: usize, seed: u64) -> Vec<f32> {
    assert!(
        nrows <= dim,
        "an orthonormal frame cannot have more rows than dimensions"
    );

    let mut rng = StableRng::new(seed);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(nrows);

    for _ in 0..nrows {
        let mut vector: Vec<f64> = (0..dim).map(|_| rng.gaussian()).collect();
        reorthogonalize(&mut vector, &basis);

        let vector_norm = norm(&vector);
        if vector_norm > 1e-12 {
            for value in &mut vector {
                *value /= vector_norm;
            }
            basis.push(vector);
        } else {
            basis.push(orthogonal_fallback(dim, &basis));
        }
    }

    basis
        .into_iter()
        .flatten()
        .map(|value| value as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn assert_orthonormal(matrix: &[f32], dim: usize, nrows: usize, tolerance: f64) {
        for i in 0..nrows {
            let row_i = &matrix[i * dim..(i + 1) * dim];
            assert!((dot(row_i, row_i) - 1.0).abs() <= tolerance);
            for j in 0..i {
                let row_j = &matrix[j * dim..(j + 1) * dim];
                assert!(
                    dot(row_i, row_j).abs() <= tolerance,
                    "rows {i} and {j} are not orthogonal: {}",
                    dot(row_i, row_j)
                );
            }
        }
    }

    #[test]
    fn high_dimensional_frames_remain_orthonormal_across_seeds() {
        for seed in [0, 1, u64::MAX] {
            let matrix = orthogonal_rotation_matrix(256, 64, seed);
            assert_orthonormal(&matrix, 256, 64, 2e-7);
        }
    }

    #[test]
    fn square_rotation_preserves_norms_and_inner_products() {
        let dim = 128;
        let matrix = orthogonal_rotation_matrix(dim, dim, 0x000A_11CE);
        let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.17).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.11).cos()).collect();
        let rotate = |vector: &[f32]| {
            matrix
                .chunks_exact(dim)
                .map(|row| dot(row, vector))
                .collect::<Vec<_>>()
        };
        let rotated_a = rotate(&a);
        let rotated_b = rotate(&b);
        let norm_a: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        let norm_rotated_a: f64 = rotated_a.iter().map(|x| x * x).sum();
        let inner: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let rotated_inner: f64 = rotated_a.iter().zip(&rotated_b).map(|(x, y)| x * y).sum();

        assert!((norm_a - norm_rotated_a).abs() <= 2e-5 * norm_a.max(1.0));
        assert!((inner - rotated_inner).abs() <= 2e-5 * inner.abs().max(1.0));
    }

    #[test]
    fn seeds_are_repeatable_and_distinct() {
        let first = orthogonal_rotation_matrix(64, 16, 42);
        let repeated = orthogonal_rotation_matrix(64, 16, 42);
        let different = orthogonal_rotation_matrix(64, 16, 43);

        assert_eq!(first, repeated);
        assert_ne!(first, different);
    }

    #[test]
    fn fallback_is_orthogonal_to_existing_basis() {
        let basis = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let fallback = orthogonal_fallback(3, &basis);
        assert!((norm(&fallback) - 1.0).abs() <= f64::EPSILON);
        assert!(
            fallback
                .iter()
                .zip(&basis[0])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs()
                <= f64::EPSILON
        );
        assert!(
            fallback
                .iter()
                .zip(&basis[1])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .abs()
                <= f64::EPSILON
        );
    }
}
