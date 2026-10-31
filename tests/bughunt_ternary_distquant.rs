//! Regressions for ternary target sparsity and the distquant Exponential prior.
#![cfg(all(feature = "ternary", feature = "distquant"))]

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[test]
fn bughunt_ternary_target_sparsity_with_default_normalize() {
    use qntz::ternary::{TernaryConfig, TernaryQuantizer};
    let dim = 128;
    let n = 500;
    let mut r = Rng(7);
    let data: Vec<f32> = (0..n * dim).map(|_| r.gauss() as f32).collect();
    let cfg = TernaryConfig {
        target_sparsity: Some(0.5),
        ..TernaryConfig::default()
    };
    assert!(cfg.normalize, "default normalizes");
    let mut q = TernaryQuantizer::new(dim, cfg);
    q.fit(&data, n).unwrap();
    let mut total = 0.0f32;
    for i in 0..n {
        total += q
            .quantize(&data[i * dim..(i + 1) * dim])
            .unwrap()
            .sparsity();
    }
    let mean = total / n as f32;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "target sparsity 0.5, observed {mean}"
    );
}

#[test]
fn bughunt_distquant_exponential_location_mismatch() {
    use qntz::distquant::{quantization_mse, quantize, Distribution};
    let mut r = Rng(99);
    let data: Vec<f64> = (0..5000).map(|_| -r.unit().ln()).collect(); // Exp(1)
    let (codes_e, info_e) = quantize(&data, 4, Distribution::Exponential, None);
    let (codes_g, info_g) = quantize(&data, 4, Distribution::Gaussian, None);
    let mse_e = quantization_mse(&data, &codes_e, &info_e);
    let mse_g = quantization_mse(&data, &codes_g, &info_g);
    let frac_code0 = codes_e.iter().filter(|&&c| c == 0).count() as f64 / data.len() as f64;
    // Before centering: mse 0.351, 67% on code 0. Equal-probability quantile
    // midpoints are not MSE-optimal, so allow 10% over the Gaussian prior
    // (0.053 here) rather than requiring a strict win.
    assert!(
        mse_e <= 1.1 * mse_g && frac_code0 < 0.2,
        "Exponential prior on Exp(1) data: mse {mse_e:.4} vs Gaussian prior {mse_g:.4}; {:.0}% of values on code 0",
        frac_code0 * 100.0
    );
}
