use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use qntz::simd_ops::{
    batch_asymmetric_l2, batch_asymmetric_l2_into, batch_hamming_distances,
    batch_hamming_distances_into, hamming_distance, pack_binary_fast, unpack_binary_fast,
};

// ============================================================================
// Helpers
// ============================================================================

fn make_f32_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_binary_codes(dim: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 63) as u8
        })
        .collect()
}

fn make_packed(dim: usize, seed: u64) -> Vec<u8> {
    let codes = make_binary_codes(dim, seed);
    let mut packed = vec![0u8; dim.div_ceil(8)];
    pack_binary_fast(&codes, &mut packed).unwrap();
    packed
}

// ============================================================================
// pack_binary_fast / unpack_binary_fast
// ============================================================================

fn bench_pack_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_binary_fast");
    for &dim in &[128usize, 768] {
        let codes = make_binary_codes(dim, 42);
        let mut packed = vec![0u8; dim.div_ceil(8)];
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                pack_binary_fast(black_box(&codes), black_box(&mut packed)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_unpack_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpack_binary_fast");
    for &dim in &[128usize, 768] {
        let packed = make_packed(dim, 42);
        let mut codes = vec![0u8; dim];
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                unpack_binary_fast(black_box(&packed), black_box(&mut codes), dim).unwrap();
            });
        });
    }
    group.finish();
}

// ============================================================================
// hamming_distance
// ============================================================================

fn bench_hamming_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamming_distance");
    for &dim in &[128usize, 768] {
        let a = make_packed(dim, 11);
        let b = make_packed(dim, 22);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b_bench, _| {
            b_bench.iter(|| hamming_distance(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_batch_distance_reuse(c: &mut Criterion) {
    let dim = 768usize;
    let n = 256usize;
    let query_bits = make_packed(dim, 11);
    let codes_bits: Vec<Vec<u8>> = (0..n).map(|i| make_packed(dim, i as u64 + 100)).collect();
    let codes_bit_refs: Vec<&[u8]> = codes_bits.iter().map(|v| v.as_slice()).collect();

    let query = make_f32_vec(dim, 22);
    let codes_l2: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let mut state = i as u64 + 500;
            (0..dim.div_ceil(8))
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 56) as u8
                })
                .collect()
        })
        .collect();
    let codes_l2_refs: Vec<&[u8]> = codes_l2.iter().map(|v| v.as_slice()).collect();

    let mut group = c.benchmark_group("batch_distance_reuse");
    group.throughput(Throughput::Elements((dim * n) as u64));
    group.bench_function("hamming_alloc_256x768", |b| {
        b.iter(|| batch_hamming_distances(black_box(&query_bits), black_box(&codes_bit_refs)));
    });
    group.bench_function("hamming_into_256x768", |b| {
        let mut out = Vec::with_capacity(n);
        b.iter(|| {
            batch_hamming_distances_into(
                black_box(&query_bits),
                black_box(&codes_bit_refs),
                black_box(&mut out),
            );
        });
    });
    group.bench_function("asymmetric_l2_alloc_256x768", |b| {
        b.iter(|| batch_asymmetric_l2(black_box(&query), black_box(&codes_l2_refs)).unwrap());
    });
    group.bench_function("asymmetric_l2_into_256x768", |b| {
        let mut out = Vec::with_capacity(n);
        b.iter(|| {
            batch_asymmetric_l2_into(
                black_box(&query),
                black_box(&codes_l2_refs),
                black_box(&mut out),
            )
            .unwrap();
        });
    });
    group.finish();
}

// ============================================================================
// RaBitQ
// ============================================================================

#[cfg(feature = "rabitq")]
fn bench_rabitq(c: &mut Criterion) {
    use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

    let mut group = c.benchmark_group("rabitq");
    for &dim in &[128usize, 768] {
        // --- 1-bit (binary) ---
        let q1 = RaBitQQuantizer::with_config(dim, 42, RaBitQConfig::binary()).unwrap();
        let doc = make_f32_vec(dim, 1);
        let query = make_f32_vec(dim, 2);
        let qv1 = q1.quantize(&doc).unwrap();
        let rotated_query1 = q1.rotate_query(&query).unwrap();

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("quantize_1bit", dim), &dim, |b, _| {
            b.iter(|| q1.quantize(black_box(&doc)).unwrap());
        });
        group.bench_with_input(
            BenchmarkId::new("approximate_l2_sqr_1bit", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    q1.approximate_l2_sqr(black_box(&query), black_box(&qv1))
                        .unwrap()
                });
            },
        );
        // Pre-rotated distance -- the hot path when rotation is amortized.
        group.bench_with_input(
            BenchmarkId::new("l2_sqr_prerotated_1bit", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    RaBitQQuantizer::approximate_l2_sqr_prerotated(
                        black_box(&rotated_query1),
                        black_box(&qv1),
                    )
                });
            },
        );

        // --- 4-bit ---
        let q4 = RaBitQQuantizer::with_config(dim, 42, RaBitQConfig::bits4()).unwrap();
        let qv4 = q4.quantize(&doc).unwrap();
        let rotated_query4 = q4.rotate_query(&query).unwrap();

        group.bench_with_input(BenchmarkId::new("quantize_4bit", dim), &dim, |b, _| {
            b.iter(|| q4.quantize(black_box(&doc)).unwrap());
        });
        group.bench_with_input(
            BenchmarkId::new("l2_sqr_prerotated_4bit", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    RaBitQQuantizer::approximate_l2_sqr_prerotated(
                        black_box(&rotated_query4),
                        black_box(&qv4),
                    )
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// TernaryQuantizer
// ============================================================================

#[cfg(feature = "ternary")]
fn bench_ternary(c: &mut Criterion) {
    use qntz::ternary::{TernaryConfig, TernaryQuantizer};

    let mut group = c.benchmark_group("ternary_quantize");
    for &dim in &[128usize, 768] {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: true,
            target_sparsity: None,
        };
        let q = TernaryQuantizer::new(dim, config);
        let v = make_f32_vec(dim, 7);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| q.quantize(black_box(&v)).unwrap());
        });
    }
    group.finish();
}

// ============================================================================
// BinaryQuantizer (rotation-based)
// ============================================================================

#[cfg(feature = "binary")]
fn bench_binary_quantizer(c: &mut Criterion) {
    use qntz::binary::BinaryQuantizer;

    let dim = 128;
    let q = BinaryQuantizer::new(dim, dim, 42);
    let v = make_f32_vec(dim, 5);
    let code = q.quantize(&v).unwrap();

    let mut group = c.benchmark_group("binary_quantizer");
    group.throughput(Throughput::Elements(dim as u64));
    group.bench_function("quantize_dim128", |b| {
        b.iter(|| q.quantize(black_box(&v)).unwrap());
    });
    group.bench_function("asymmetric_distance_dim128", |b| {
        b.iter(|| {
            q.asymmetric_distance(black_box(&v), black_box(&code))
                .unwrap()
        });
    });
    group.finish();
}

// ============================================================================
// AdaptiveQuantizer
// ============================================================================

#[cfg(feature = "adaptive")]
fn bench_adaptive(c: &mut Criterion) {
    use qntz::adaptive::AdaptiveQuantizer;

    let dim = 128;
    let n = 64;
    let q = AdaptiveQuantizer::new(8).unwrap();
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_f32_vec(dim, i as u64)).collect();
    let query = make_f32_vec(dim, 999);
    let packed = q.quantize_packed(&vectors).unwrap();

    let mut group = c.benchmark_group("adaptive_quantizer");
    group.throughput(Throughput::Elements((dim * n) as u64));
    group.bench_function("quantize_packed_64x128", |b| {
        b.iter(|| q.quantize_packed(black_box(&vectors)).unwrap());
    });
    group.bench_function("asymmetric_distances_64x128", |b| {
        b.iter(|| packed.asymmetric_distances(black_box(&query)).unwrap());
    });
    group.bench_function("asymmetric_distances_into_64x128", |b| {
        let mut out = Vec::with_capacity(n);
        b.iter(|| {
            packed
                .asymmetric_distances_into(black_box(&query), black_box(&mut out))
                .unwrap();
        });
    });
    group.finish();
}

// ============================================================================
// Benchmark registration
// ============================================================================

fn base_benches(c: &mut Criterion) {
    bench_pack_binary(c);
    bench_unpack_binary(c);
    bench_hamming_distance(c);
    bench_batch_distance_reuse(c);
    #[cfg(feature = "rabitq")]
    bench_rabitq(c);
    #[cfg(feature = "ternary")]
    bench_ternary(c);
    #[cfg(feature = "binary")]
    bench_binary_quantizer(c);
    #[cfg(feature = "adaptive")]
    bench_adaptive(c);
}

criterion_group!(benches, base_benches);
criterion_main!(benches);
