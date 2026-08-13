#![cfg(feature = "rabitq")]
#![allow(clippy::unwrap_used)]
//! Tests that RaBitQ approximate distances correlate with true L2 distances.
//!
//! These tests measure the rank correlation (Spearman) between approximate
//! and exact distances. A broken distance function will have near-zero
//! correlation, while a working one should have > 0.5 even at low dimensions.

use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut h);
                    i.hash(&mut h);
                    j.hash(&mut h);
                    (h.finish() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                })
                .collect();
            normalize(&v)
        })
        .collect()
}

/// Spearman rank correlation between two vectors.
fn spearman_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 3 {
        return 0.0;
    }

    fn rank(vals: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut ranks = vec![0.0f32; vals.len()];
        for (rank, &(idx, _)) in indexed.iter().enumerate() {
            ranks[idx] = rank as f32;
        }
        ranks
    }

    let ra = rank(a);
    let rb = rank(b);

    let mean_a: f32 = ra.iter().sum::<f32>() / n as f32;
    let mean_b: f32 = rb.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        cov / denom
    }
}

/// Test distance correlation for a given bit config and dimension.
fn test_distance_correlation(total_bits: usize, dim: usize, expected_min_corr: f32) {
    let config = RaBitQConfig {
        total_bits,
        t_const: None,
    };
    let n_data = 100;
    let n_queries = 10;

    let mut quantizer = RaBitQQuantizer::with_config(dim, 42, config).unwrap();
    let data = random_vectors(n_data, dim, 77);
    let queries = random_vectors(n_queries, dim, 1337);

    // Fit centroid
    let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
    quantizer.fit(&flat, n_data).unwrap();

    // Quantize all data vectors
    let quantized: Vec<_> = data
        .iter()
        .map(|v| quantizer.quantize(v).unwrap())
        .collect();

    // For each query, compute exact and approximate distances to all data vectors
    let mut correlations = Vec::new();
    for q in &queries {
        let exact: Vec<f32> = data.iter().map(|d| l2_sqr(q, d)).collect();
        let approx: Vec<f32> = quantized
            .iter()
            .map(|qv| quantizer.approximate_l2_sqr(q, qv).unwrap())
            .collect();

        let corr = spearman_correlation(&exact, &approx);
        correlations.push(corr);
    }

    let mean_corr: f32 = correlations.iter().sum::<f32>() / correlations.len() as f32;

    assert!(
        mean_corr >= expected_min_corr,
        "{}bit dim={}: mean Spearman correlation {:.3} < {:.3} (per-query: {:?})",
        total_bits,
        dim,
        mean_corr,
        expected_min_corr,
        correlations
            .iter()
            .map(|c| format!("{:.2}", c))
            .collect::<Vec<_>>()
    );
}

// -------------------------------------------------------------------
// Binary (1-bit) -- baseline, known to work
// -------------------------------------------------------------------

#[test]
fn binary_correlation_dim32() {
    test_distance_correlation(1, 32, 0.5);
}

#[test]
fn binary_correlation_dim64() {
    test_distance_correlation(1, 64, 0.6);
}

#[test]
fn binary_correlation_dim128() {
    test_distance_correlation(1, 128, 0.7);
}

// -------------------------------------------------------------------
// 4-bit -- should be BETTER than binary, not worse
// -------------------------------------------------------------------

#[test]
fn four_bit_correlation_dim32() {
    // 4-bit should correlate at least as well as 1-bit
    test_distance_correlation(4, 32, 0.5);
}

#[test]
fn four_bit_correlation_dim64() {
    test_distance_correlation(4, 64, 0.6);
}

#[test]
fn four_bit_correlation_dim128() {
    test_distance_correlation(4, 128, 0.7);
}

// -------------------------------------------------------------------
// 8-bit -- should be BETTER than 4-bit
// -------------------------------------------------------------------

#[test]
fn eight_bit_correlation_dim32() {
    test_distance_correlation(8, 32, 0.5);
}

#[test]
fn eight_bit_correlation_dim64() {
    test_distance_correlation(8, 64, 0.6);
}

#[test]
fn eight_bit_correlation_dim128() {
    test_distance_correlation(8, 128, 0.7);
}

// -------------------------------------------------------------------
// Monotonicity: more bits should give better correlation
// -------------------------------------------------------------------

#[test]
fn more_bits_should_improve_correlation() {
    let dim = 64;
    let config_1 = RaBitQConfig {
        total_bits: 1,
        t_const: None,
    };
    let config_4 = RaBitQConfig {
        total_bits: 4,
        t_const: None,
    };

    let n_data = 200;
    let mut q1 = RaBitQQuantizer::with_config(dim, 42, config_1).unwrap();
    let mut q4 = RaBitQQuantizer::with_config(dim, 42, config_4).unwrap();
    let data = random_vectors(n_data, dim, 77);
    let queries = random_vectors(20, dim, 1337);
    let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
    q1.fit(&flat, n_data).unwrap();
    q4.fit(&flat, n_data).unwrap();

    let qv1: Vec<_> = data.iter().map(|v| q1.quantize(v).unwrap()).collect();
    let qv4: Vec<_> = data.iter().map(|v| q4.quantize(v).unwrap()).collect();

    let mut corr1_total = 0.0f32;
    let mut corr4_total = 0.0f32;
    for q in &queries {
        let exact: Vec<f32> = data.iter().map(|d| l2_sqr(q, d)).collect();
        let approx1: Vec<f32> = qv1
            .iter()
            .map(|qv| q1.approximate_l2_sqr(q, qv).unwrap())
            .collect();
        let approx4: Vec<f32> = qv4
            .iter()
            .map(|qv| q4.approximate_l2_sqr(q, qv).unwrap())
            .collect();
        corr1_total += spearman_correlation(&exact, &approx1);
        corr4_total += spearman_correlation(&exact, &approx4);
    }
    let avg1 = corr1_total / queries.len() as f32;
    let avg4 = corr4_total / queries.len() as f32;

    assert!(
        avg4 >= avg1 - 0.05,
        "4-bit correlation ({:.3}) should be >= 1-bit ({:.3})",
        avg4,
        avg1
    );
}

#[test]
fn error_margin_scales_with_query_residual_norm() {
    let dim = 32;
    let mut quantizer = RaBitQQuantizer::binary(dim, 42).unwrap();
    let centroid = vec![3.0f32; dim];
    quantizer.set_centroid(centroid.clone()).unwrap();
    let target: Vec<f32> = centroid
        .iter()
        .enumerate()
        .map(|(i, c)| c + (i as f32 + 1.0).recip())
        .collect();
    let quantized = quantizer.quantize(&target).unwrap();

    let zero = quantizer
        .squared_distance_error_margin(&centroid, &quantized)
        .unwrap();
    let query: Vec<f32> = centroid
        .iter()
        .enumerate()
        .map(|(i, c)| c + if i == 0 { 1.0 } else { 0.0 })
        .collect();
    let doubled: Vec<f32> = centroid
        .iter()
        .enumerate()
        .map(|(i, c)| c + if i == 0 { 2.0 } else { 0.0 })
        .collect();
    let margin = quantizer
        .squared_distance_error_margin(&query, &quantized)
        .unwrap();
    let doubled_margin = quantizer
        .squared_distance_error_margin(&doubled, &quantized)
        .unwrap();

    assert_eq!(zero, 0.0);
    assert!(margin.is_finite() && margin > 0.0);
    assert!((doubled_margin - 2.0 * margin).abs() <= 1e-6 * margin.max(1.0));
}

#[test]
fn binary_error_margin_covers_adversarial_finite_pairs() {
    let dim = 64;
    let basis = |index: usize| {
        let mut vector = vec![0.0f32; dim];
        vector[index] = 1.0;
        vector
    };
    let dense = normalize(&(0..dim).map(|i| i as f32 + 1.0).collect::<Vec<_>>());
    let alternating = normalize(
        &(0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>(),
    );
    let mut spiked = vec![1e-3f32; dim];
    spiked[0] = 1.0;
    let spiked = normalize(&spiked);
    let pairs = [
        (basis(0), basis(1)),
        (basis(0), dense.clone()),
        (alternating, dense),
        (spiked, basis(dim - 1)),
    ];

    let mut covered = 0usize;
    let mut total = 0usize;
    for seed in 1..=128 {
        let quantizer = RaBitQQuantizer::binary(dim, seed).unwrap();
        for (target, query) in &pairs {
            let code = quantizer.quantize(target).unwrap();
            let estimate = quantizer.approximate_l2_sqr(query, &code).unwrap();
            let exact = l2_sqr(query, target);
            let margin = quantizer
                .squared_distance_error_margin(query, &code)
                .unwrap();
            assert!(estimate.is_finite() && margin.is_finite());
            covered += usize::from((estimate - exact).abs() <= margin + 1e-6);
            total += 1;
        }
    }

    let coverage = covered as f32 / total as f32;
    assert!(
        coverage >= 0.95,
        "epsilon=1.9 margin covered only {covered}/{total} adversarial pairs ({:.1}%)",
        coverage * 100.0
    );
}
