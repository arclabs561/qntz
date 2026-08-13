//! Deterministic, on-demand statistical evaluation of the RaBitQ estimator.
//!
//! The default run is a bounded smoke test. Pass `--full` for more dimensions
//! and rotation seeds. This executable deliberately avoids timing claims.

use qntz::rabitq::{QuantizedVector, RaBitQQuantizer};

trait EvaluatedRotator {
    type Code;

    fn name(&self) -> &'static str;
    fn dimension(&self) -> usize;
    fn rotate(&self, vector: &[f32]) -> qntz::Result<Vec<f32>>;
    fn encode(&self, vector: &[f32]) -> qntz::Result<Self::Code>;
    fn estimate_unclamped(&self, rotated_query: &[f32], code: &Self::Code) -> f32;
    fn error_margin(&self, query: &[f32], code: &Self::Code) -> qntz::Result<f32>;
}

struct DenseRaBitQ {
    inner: RaBitQQuantizer,
    dimension: usize,
}

impl DenseRaBitQ {
    fn new(dimension: usize, seed: u64) -> qntz::Result<Self> {
        Ok(Self {
            inner: RaBitQQuantizer::binary(dimension, seed)?,
            dimension,
        })
    }
}

impl EvaluatedRotator for DenseRaBitQ {
    type Code = QuantizedVector;

    fn name(&self) -> &'static str {
        "dense"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn rotate(&self, vector: &[f32]) -> qntz::Result<Vec<f32>> {
        self.inner.rotate_query(vector)
    }

    fn encode(&self, vector: &[f32]) -> qntz::Result<Self::Code> {
        self.inner.quantize(vector)
    }

    fn estimate_unclamped(&self, rotated_query: &[f32], code: &Self::Code) -> f32 {
        RaBitQQuantizer::approximate_l2_sqr_prerotated(rotated_query, code)
    }

    fn error_margin(&self, query: &[f32], code: &Self::Code) -> qntz::Result<f32> {
        self.inner.squared_distance_error_margin(query, code)
    }
}

#[derive(Clone, Copy)]
enum Family {
    Random,
    Basis,
    Sparse,
    Spiked,
    Alternating,
    Correlated,
    NearCollinear,
}

impl Family {
    const ALL: [Self; 7] = [
        Self::Random,
        Self::Basis,
        Self::Sparse,
        Self::Spiked,
        Self::Alternating,
        Self::Correlated,
        Self::NearCollinear,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Basis => "basis",
            Self::Sparse => "sparse",
            Self::Spiked => "spiked",
            Self::Alternating => "alternating",
            Self::Correlated => "correlated",
            Self::NearCollinear => "near-collinear",
        }
    }
}

#[derive(Default)]
struct Metrics {
    errors: Vec<f64>,
    covered: usize,
}

impl Metrics {
    fn record(&mut self, error: f64, margin: f64) {
        self.errors.push(error);
        self.covered += usize::from(error.abs() <= margin);
    }

    fn report(&mut self) -> (f64, f64, f64, f64, f64) {
        self.errors.sort_by(f64::total_cmp);
        let n = self.errors.len() as f64;
        let bias = self.errors.iter().sum::<f64>() / n;
        let rmse = (self.errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
        let abs: Vec<f64> = self.errors.iter().map(|e| e.abs()).collect();
        let p95 = percentile(&abs, 0.95);
        let p99 = percentile(&abs, 0.99);
        let coverage = self.covered as f64 / n;
        (bias, rmse, p95, p99, coverage)
    }
}

fn main() -> qntz::Result<()> {
    let full = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--full") => true,
        Some(flag) => {
            eprintln!("unknown argument: {flag}; expected --full");
            std::process::exit(2);
        }
    };
    let dimensions: &[usize] = if full { &[32, 64, 128] } else { &[32] };
    let seeds = if full { 64 } else { 8 };
    let pairs_per_family = if full { 16 } else { 4 };

    println!(
        "mode={} seeds={seeds} pairs/family={pairs_per_family}",
        if full { "full" } else { "smoke" }
    );
    println!(
        "rotator dim family             n       bias       rmse      p95|e|    p99|e| coverage"
    );

    for &dimension in dimensions {
        let mut by_family: Vec<Metrics> = Family::ALL.iter().map(|_| Metrics::default()).collect();
        let mut worst_orthogonality = 0.0f64;
        let rotator_name = DenseRaBitQ::new(dimension, 0)?.name();
        for seed in 0..seeds {
            let rotator = DenseRaBitQ::new(dimension, seed as u64)?;
            worst_orthogonality = worst_orthogonality.max(orthogonality_error(&rotator)?);
            let mut rng = Rng::new((seed as u64) ^ ((dimension as u64) << 32));
            for (family_index, family) in Family::ALL.iter().copied().enumerate() {
                for sample in 0..pairs_per_family {
                    let (target, query) = pair(family, dimension, sample, &mut rng);
                    let code = rotator.encode(&target)?;
                    let rotated_query = rotator.rotate(&query)?;
                    let estimate = rotator.estimate_unclamped(&rotated_query, &code);
                    let exact_proxy = l2_sqr(&query, &target) - norm_sqr(&query);
                    let margin = rotator.error_margin(&query, &code)?;
                    by_family[family_index]
                        .record(f64::from(estimate - exact_proxy), f64::from(margin));
                }
            }
        }

        for (family, metrics) in Family::ALL.iter().zip(&mut by_family) {
            let (bias, rmse, p95, p99, coverage) = metrics.report();
            let coverage_pct = coverage * 100.0;
            println!(
                "{:<7} {:>3} {:<16} {:>5} {bias:>10.5} {rmse:>10.5} {p95:>10.5} {p99:>10.5} {coverage_pct:>7.1}%",
                rotator_name,
                dimension,
                family.name(),
                metrics.errors.len(),
            );
        }
        println!(
            "{rotator_name:<7} {dimension:>3} rotation max |R^T R - I| = {worst_orthogonality:.3e}"
        );
    }
    Ok(())
}

fn orthogonality_error(rotator: &impl EvaluatedRotator) -> qntz::Result<f64> {
    let dim = rotator.dimension();
    let mut columns = Vec::with_capacity(dim);
    for index in 0..dim {
        let mut basis = vec![0.0f32; dim];
        basis[index] = 1.0;
        columns.push(rotator.rotate(&basis)?);
    }
    let mut worst = 0.0f64;
    for i in 0..dim {
        for j in 0..=i {
            let actual: f64 = columns[i]
                .iter()
                .zip(&columns[j])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((actual - expected).abs());
        }
    }
    Ok(worst)
}

fn pair(family: Family, dim: usize, sample: usize, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    match family {
        Family::Random => (unit_random(dim, rng), unit_random(dim, rng)),
        Family::Basis => (basis(dim, sample % dim), basis(dim, (sample + 1) % dim)),
        Family::Sparse => {
            let mut a = vec![0.0; dim];
            let mut b = vec![0.0; dim];
            for k in 0..4.min(dim) {
                a[(sample + k * 7) % dim] = if k % 2 == 0 { 1.0 } else { -0.5 };
                b[(sample + k * 11 + 1) % dim] = if k % 2 == 0 { -0.7 } else { 0.4 };
            }
            (normalize(a), normalize(b))
        }
        Family::Spiked => {
            let mut a = vec![1e-3; dim];
            let mut b = vec![-1e-3; dim];
            a[sample % dim] = 1.0;
            b[(sample + 1) % dim] = 1.0;
            (normalize(a), normalize(b))
        }
        Family::Alternating => {
            let a = (0..dim)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let b = (0..dim)
                .map(|i| if i % 4 < 2 { 1.0 } else { -1.0 })
                .collect();
            (normalize(a), normalize(b))
        }
        Family::Correlated => {
            let base = unit_random(dim, rng);
            let noise = unit_random(dim, rng);
            let query = normalize(base.iter().zip(noise).map(|(&x, n)| x + 0.2 * n).collect());
            (base, query)
        }
        Family::NearCollinear => {
            let base = unit_random(dim, rng);
            let noise = unit_random(dim, rng);
            let sign = if sample % 2 == 0 { 1.0 } else { -1.0 };
            let query = normalize(
                base.iter()
                    .zip(noise)
                    .map(|(&x, n)| sign * x + 1e-3 * n)
                    .collect(),
            );
            (base, query)
        }
    }
}

fn basis(dim: usize, index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[index] = 1.0;
    vector
}

fn unit_random(dim: usize, rng: &mut Rng) -> Vec<f32> {
    normalize((0..dim).map(|_| rng.normal()).collect())
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = norm_sqr(&vector).sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn norm_sqr(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x * x).sum()
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    sorted[index]
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        (((self.next() >> 40) as f32) + 0.5) / ((1u32 << 24) as f32)
    }

    fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * self.uniform();
        radius * angle.cos()
    }
}
