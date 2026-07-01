//! RaBitQ recall vs compression on SIFT-small, with filter-then-rerank.
//!
//! Quantize 10k 128-dim base vectors, rank each query by the cheap approximate
//! distance, rerank the top-C candidates by exact L2. Reports recall@10 against
//! code budget; C=10 is the no-rerank baseline.
//!
//! ```sh
//! ./scripts/fetch_siftsmall.sh
//! cargo run --release --features rabitq --example sift_rabitq_recall
//! ```

use std::path::Path;
use std::process::ExitCode;

use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

const DIM: usize = 128;
const K: usize = 10;

/// Read an `.fvecs` file: each record is `int32 dim` then `dim` little-endian f32.
fn read_fvecs(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut o = 0;
    while o < bytes.len() {
        let dim = i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        o += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(f32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
            o += 4;
        }
        out.push(v);
    }
    Ok(out)
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Indices of the k smallest distances.
fn top_k(dists: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..dists.len()).collect();
    idx.sort_unstable_by(|&a, &b| dists[a].total_cmp(&dists[b]));
    idx.truncate(k);
    idx
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/siftsmall");
    let base_path = dir.join("siftsmall_base.fvecs");
    let query_path = dir.join("siftsmall_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        println!(
            "dataset not found at {}\nrun: ./scripts/fetch_siftsmall.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let base = read_fvecs(&base_path).unwrap();
    let queries = read_fvecs(&query_path).unwrap();
    let n = base.len();
    println!("base: {n} x {DIM}   queries: {}   k = {K}", queries.len());

    // Exact L2 top-k per query (ground truth for recall).
    let exact: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let dists: Vec<f32> = base.iter().map(|b| l2_sqr(q, b)).collect();
            top_k(&dists, K)
        })
        .collect();

    let base_flat: Vec<f32> = base.iter().flatten().copied().collect();
    let float_bytes = DIM * 4;
    // Candidate budgets for the rerank stage. C=10 == no rerank (read top-10
    // straight off the codes), the rest filter C by code then rerank by exact.
    let budgets = [10usize, 50, 100, 500];

    println!("\nfloat32 baseline: {float_bytes} bytes/vector");
    println!("recall@{K} at candidate budget C (C=10 is no-rerank):\n");
    print!("  {:>6}  {:>10}  {:>6}", "bits", "bytes/vec", "ratio");
    for c in budgets {
        print!("   C={c:<4}");
    }
    println!();

    for (bits, cfg) in [
        (1u32, RaBitQConfig::binary()),
        (2, RaBitQConfig::bits2()),
        (4, RaBitQConfig::bits4()),
        (8, RaBitQConfig::bits8()),
    ] {
        let mut quant = RaBitQQuantizer::with_config(DIM, 42, cfg).unwrap();
        quant.fit(&base_flat, n).unwrap();
        let codes: Vec<_> = base.iter().map(|v| quant.quantize(v).unwrap()).collect();

        let mut recall_sums = vec![0.0f64; budgets.len()];
        for (q, exact_ids) in queries.iter().zip(&exact) {
            let approx_dists: Vec<f32> = codes
                .iter()
                .map(|c| quant.approximate_l2_sqr(q, c).unwrap())
                .collect();
            for (bi, &c) in budgets.iter().enumerate() {
                // Filter: top-C by cheap approximate distance.
                let candidates = top_k(&approx_dists, c);
                // Rerank: exact L2 on the candidate float vectors, keep top-K.
                let mut reranked: Vec<(usize, f32)> = candidates
                    .iter()
                    .map(|&i| (i, l2_sqr(q, &base[i])))
                    .collect();
                reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let hits = reranked
                    .iter()
                    .take(K)
                    .filter(|(i, _)| exact_ids.contains(i))
                    .count();
                recall_sums[bi] += hits as f64 / K as f64;
            }
        }

        let code_bytes = (bits as usize * DIM).div_ceil(8);
        print!(
            "  {bits:>6}  {code_bytes:>10}  {:>5.0}x",
            float_bytes as f64 / code_bytes as f64
        );
        for sum in &recall_sums {
            print!("  {:>6.3}", sum / queries.len() as f64);
        }
        println!();
    }

    ExitCode::SUCCESS
}
