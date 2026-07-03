use qntz::adaptive::AdaptiveQuantizer;

const DIM: usize = 96;
const N_DOCS: usize = 256;
const TOP_K: usize = 10;

fn main() -> qntz::Result<()> {
    let docs = make_vectors(N_DOCS, DIM);
    let query = make_query(&docs[37]);
    let exact = exact_top_k(&docs, &query, TOP_K);

    println!("dataset: {N_DOCS} docs, dim={DIM}, top-{TOP_K}");
    println!("bits  recall@10  mean distance relative error  stored code bytes/doc");

    let mut distances = Vec::with_capacity(N_DOCS);
    for bits in [2u8, 4, 8] {
        let quantizer = AdaptiveQuantizer::new(bits)?;
        let batch = quantizer.quantize_packed(&docs)?;
        let scan_plan = batch.scan_plan();
        scan_plan.asymmetric_distances_into(&query, &mut distances)?;
        let approx = top_k_from_distances(&distances, TOP_K);

        let recall = recall_at_k(&exact, &approx);
        let mean_rel_err = mean_relative_error(&docs, &query, &distances);
        let bytes_per_doc = batch.codes.len() / batch.len();

        println!("{bits:>4}  {recall:>8.3}  {mean_rel_err:>28.4}  {bytes_per_doc:>9}");
    }

    Ok(())
}

fn top_k_from_distances(distances: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = distances.iter().copied().enumerate().collect();
    scores.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

fn exact_top_k(docs: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(doc_id, doc)| (doc_id, l2_sqr(query, doc)))
        .collect();
    scores.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

fn recall_at_k(exact: &[(usize, f32)], approx: &[(usize, f32)]) -> f32 {
    let hits = approx
        .iter()
        .filter(|(doc_id, _)| exact.iter().any(|(exact_id, _)| exact_id == doc_id))
        .count();
    hits as f32 / exact.len() as f32
}

fn mean_relative_error(docs: &[Vec<f32>], query: &[f32], approx_distances: &[f32]) -> f32 {
    let total: f32 = docs
        .iter()
        .zip(approx_distances.iter())
        .map(|(doc, &approx)| {
            let exact = l2_sqr(query, doc).max(1e-6);
            ((approx - exact).abs() / exact).min(9.99)
        })
        .sum();
    total / docs.len() as f32
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(0x1234_5678);
    (0..n)
        .map(|i| {
            let scale = 0.5 + (i % 11) as f32 * 0.2;
            let offset = (i % 7) as f32 * 0.15 - 0.45;
            (0..dim)
                .map(|d| {
                    let periodic = ((d as f32 + i as f32 * 0.03) * 0.11).sin();
                    offset + scale * (periodic + 0.2 * rng.normal())
                })
                .collect()
        })
        .collect()
}

fn make_query(source: &[f32]) -> Vec<f32> {
    let mut rng = Rng::new(0xCAFE);
    source.iter().map(|&x| x + 0.05 * rng.normal()).collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
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
