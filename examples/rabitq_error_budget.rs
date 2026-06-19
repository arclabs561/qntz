use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

const DIM: usize = 64;
const N_DOCS: usize = 512;

fn main() -> qntz::Result<()> {
    let docs = make_clustered_vectors(N_DOCS, DIM, 0xA11CE);
    let flat_docs: Vec<f32> = docs.iter().flat_map(|v| v.iter().copied()).collect();

    println!("dataset: {N_DOCS} docs, dim={DIM}");
    println!("bits  mean relative error proxy  mean residual norm  bytes/code");

    for bits in [1usize, 2, 4, 8] {
        let config = RaBitQConfig {
            total_bits: bits,
            t_const: None,
        };
        let mut quantizer = RaBitQQuantizer::with_config(DIM, 42, config)?;
        quantizer.fit(&flat_docs, N_DOCS)?;

        let mut rel_error_sum = 0.0f32;
        let mut residual_sum = 0.0f32;
        let mut bytes_sum = 0usize;

        for doc in &docs {
            let qv = quantizer.quantize(doc)?;
            let denom = (qv.residual_norm * qv.residual_norm).max(1e-6);
            rel_error_sum += qv.f_error / denom;
            residual_sum += qv.residual_norm;
            bytes_sum += qv.binary_codes.len() + qv.extended_codes.len();
        }

        let n = docs.len() as f32;
        let mean_rel_error = rel_error_sum / n;
        let mean_residual = residual_sum / n;
        let bytes_per_code = bytes_sum as f32 / n;

        println!(
            "{bits:>4}  {mean_rel_error:>25.4}  {mean_residual:>18.4}  {bytes_per_code:>10.1}"
        );
    }

    Ok(())
}

fn make_clustered_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..dim).map(|_| rng.normal() * 0.8).collect())
        .collect();

    (0..n)
        .map(|i| {
            let center = &centers[i % centers.len()];
            center.iter().map(|&c| c + 0.18 * rng.normal()).collect()
        })
        .collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
