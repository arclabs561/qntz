//! Compare RaBitQ candidates with exact L2 neighbors on two vector files.
//!
//! ```sh
//! cargo run --release --features rabitq --example sift_rabitq_recall -- base.fvecs queries.fvecs
//! ```
//!
//! With no arguments, read the files installed by `scripts/fetch_siftsmall.sh`.

use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

fn parse_fvecs(bytes: &[u8]) -> io::Result<Vec<Vec<f32>>> {
    let invalid = |message| io::Error::new(ErrorKind::InvalidData, message);
    let mut input = bytes;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    while !input.is_empty() {
        let header = input
            .get(..4)
            .ok_or_else(|| invalid("truncated dimension"))?;
        let dimension = i32::from_le_bytes(header.try_into().unwrap());
        if dimension <= 0 {
            return Err(invalid("dimension must be positive"));
        }
        let dimension = dimension as usize;
        if rows.first().is_some_and(|row| row.len() != dimension) {
            return Err(invalid("row dimensions differ"));
        }
        let size = dimension
            .checked_mul(4)
            .ok_or_else(|| invalid("dimension overflow"))?;
        input = &input[4..];
        let payload = input.get(..size).ok_or_else(|| invalid("truncated row"))?;
        let row: Vec<_> = payload
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        if !row.iter().all(|v| v.is_finite()) {
            return Err(invalid("non-finite coordinate"));
        }
        rows.push(row);
        input = &input[size..];
    }
    if rows.is_empty() {
        return Err(invalid("empty vector file"));
    }
    Ok(rows)
}

fn read_fvecs(path: &Path) -> io::Result<Vec<Vec<f32>>> {
    std::fs::read(path)
        .and_then(|bytes| parse_fvecs(&bytes))
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Indices of the k smallest distances.
fn top_k(dists: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..dists.len()).collect();
    idx.sort_unstable_by(|&a, &b| dists[a].total_cmp(&dists[b]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let (base_path, query_path) = match args.as_slice() {
        [] => {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/siftsmall");
            let base = dir.join("siftsmall_base.fvecs");
            let queries = dir.join("siftsmall_query.fvecs");
            if !base.exists() || !queries.exists() {
                println!(
                    "dataset not found at {}\nrun: ./scripts/fetch_siftsmall.sh",
                    dir.display()
                );
                return Ok(());
            }
            (base, queries)
        }
        [base, queries] => (base.clone(), queries.clone()),
        _ => return Err("usage: sift_rabitq_recall [BASE.fvecs QUERIES.fvecs]".into()),
    };
    let base = read_fvecs(&base_path)?;
    let queries = read_fvecs(&query_path)?;
    let dim = base[0].len();
    if dim != queries[0].len() {
        return Err(format!(
            "base dimension {dim} differs from query dimension {}",
            queries[0].len()
        )
        .into());
    }
    let n = base.len();
    let k = n.min(10);
    println!("base: {n} x {dim}   queries: {}   k = {k}", queries.len());

    // Exact L2 top-k per query (ground truth for recall).
    let exact: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let dists: Vec<f32> = base.iter().map(|b| l2_sqr(q, b)).collect();
            top_k(&dists, k)
        })
        .collect();

    let base_flat: Vec<f32> = base.iter().flatten().copied().collect();
    let float_bytes = dim * 4;
    // Candidate budgets for the rerank stage. C=10 == no rerank (read top-10
    // straight off the codes), the rest filter C by code then rerank by exact.
    let mut budgets = [10usize, 50, 100, 500].map(|c| c.min(n)).to_vec();
    budgets.dedup();

    println!("\nfloat32 baseline: {float_bytes} bytes/vector");
    println!("recall@{k} after exact reranking of C candidates:\n");
    print!(
        "  {:>6}  {:>13}  {:>12}",
        "bits", "payload bytes", "record bytes"
    );
    for c in &budgets {
        print!("   C={c:<4}");
    }
    println!();

    for (bits, cfg) in [
        (1u32, RaBitQConfig::binary()),
        (2, RaBitQConfig::bits2()),
        (4, RaBitQConfig::bits4()),
        (8, RaBitQConfig::bits8()),
    ] {
        let mut quant = RaBitQQuantizer::with_config(dim, 42, cfg)?;
        quant.fit(&base_flat, n)?;
        let codes: Vec<_> = base
            .iter()
            .map(|v| quant.quantize(v))
            .collect::<qntz::Result<_>>()?;

        let mut recall_sums = vec![0.0f64; budgets.len()];
        for (q, exact_ids) in queries.iter().zip(&exact) {
            // The query-only distance offset cancels when ranking candidates.
            let rotated = quant.rotate_query(q)?;
            let approx_dists: Vec<f32> = codes
                .iter()
                .map(|c| RaBitQQuantizer::try_approximate_l2_sqr_prerotated(&rotated, c))
                .collect::<qntz::Result<_>>()?;
            for (bi, &c) in budgets.iter().enumerate() {
                // Filter: top-C by cheap approximate distance.
                let candidates = top_k(&approx_dists, c);
                // Rerank: exact L2 on the candidate float vectors, keep top-K.
                let mut reranked: Vec<(usize, f32)> = candidates
                    .iter()
                    .map(|&i| (i, l2_sqr(q, &base[i])))
                    .collect();
                reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                let hits = reranked
                    .iter()
                    .take(k)
                    .filter(|(i, _)| exact_ids.contains(i))
                    .count();
                recall_sums[bi] += hits as f64 / k as f64;
            }
        }

        let code = &codes[0];
        let payload = code.binary_codes.len() + code.extended_codes.len();
        print!("  {bits:>6}  {payload:>13}  {:>12}", code.encoded_len());
        for sum in &recall_sums {
            print!("  {:>6.3}", sum / queries.len() as f64);
        }
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(values: &[f32]) -> Vec<u8> {
        let mut bytes = (values.len() as i32).to_le_bytes().to_vec();
        bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    #[test]
    fn reads_multiple_vectors_and_rejects_invalid_rows() {
        let row = record(&[1.0, 2.0]);
        assert_eq!(
            parse_fvecs(&row.repeat(2)).unwrap(),
            vec![vec![1.0, 2.0]; 2]
        );
        for invalid in [
            vec![],
            vec![0, 0],
            record(&[]),
            (-1i32).to_le_bytes().to_vec(),
            row[..row.len() - 1].to_vec(),
            [row, record(&[3.0])].concat(),
            record(&[f32::NAN]),
        ] {
            assert!(parse_fvecs(&invalid).is_err());
        }
    }

    #[test]
    fn ties_use_original_row_order() {
        assert_eq!(top_k(&[2.0, 1.0, 1.0, 1.0], 2), vec![1, 2]);
    }
}
