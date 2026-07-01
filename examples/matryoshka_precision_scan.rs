//! Scan the same Matryoshka codes at multiple bit precisions.
//!
//! Documents are quantized once at 8 bits. At query time, the example slices
//! those stored codes to 2, 4, and 8 bits, reconstructs on the shared grid, and
//! compares the resulting scan against exact L2.
//!
//! The example compares ordinary nearest 8-bit parent codes against an
//! example-local joint parent-code selector. The joint selector chooses the
//! parent code whose sliced reconstructions minimize a weighted error across
//! the target precisions.
//!
//! Run: cargo run --release --features matryoshka --example matryoshka_precision_scan

use qntz::matryoshka::MatryoshkaQuantizer;
use std::collections::HashSet;

const N_DOCS: usize = 256;
const DIM: usize = 64;
const TOP_K: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = make_docs();
    let query = make_query(&docs[37]);
    let values: Vec<f32> = docs.iter().flat_map(|doc| doc.iter().copied()).collect();
    let quantizer = MatryoshkaQuantizer::fit(&values, 8, &[8, 4, 2], &[0.2, 0.3, 1.0])?;
    let exact_distances: Vec<f32> = docs.iter().map(|doc| l2_sqr(&query, doc)).collect();
    let exact_top = top_k_from_distances(&exact_distances, TOP_K);

    println!("dataset: {N_DOCS} docs, dim={DIM}, top-{TOP_K}");
    println!("one stored 8-bit code per scalar, sliced at query time\n");
    println!("mode     bits  recall@10  mean distance relative error  stored bytes/doc");

    for mode in [ParentCodeMode::Nearest, ParentCodeMode::Joint] {
        let codes = make_codes(&docs, &quantizer, mode);
        let mut rows = Vec::new();

        for bits in [2u8, 4, 8] {
            let approx_distances: Vec<f32> = codes
                .iter()
                .map(|doc_codes| sliced_l2_sqr(&query, doc_codes, &quantizer, bits))
                .collect();
            let approx_top = top_k_from_distances(&approx_distances, TOP_K);
            let recall = recall_at_k(&exact_top, &approx_top);
            let mean_rel_err = mean_relative_error(&exact_distances, &approx_distances);
            let stored_bytes = codes[0].len();

            println!(
                "{mode:>7}  {bits:>4}  {recall:>8.3}  {mean_rel_err:>28.4}  {stored_bytes:>16}",
                mode = mode.label()
            );
            rows.push((bits, recall, mean_rel_err));
        }

        let err2 = rows.iter().find(|(bits, _, _)| *bits == 2).unwrap().2;
        let err4 = rows.iter().find(|(bits, _, _)| *bits == 4).unwrap().2;
        let err8 = rows.iter().find(|(bits, _, _)| *bits == 8).unwrap().2;
        let recall2 = rows.iter().find(|(bits, _, _)| *bits == 2).unwrap().1;
        let recall8 = rows.iter().find(|(bits, _, _)| *bits == 8).unwrap().1;

        assert!(err8 < err4 && err4 < err2);
        assert!(recall8 >= recall2);
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum ParentCodeMode {
    Nearest,
    Joint,
}

impl ParentCodeMode {
    fn label(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Joint => "joint",
        }
    }
}

fn make_codes(
    docs: &[Vec<f32>],
    quantizer: &MatryoshkaQuantizer,
    mode: ParentCodeMode,
) -> Vec<Vec<u8>> {
    docs.iter()
        .map(|doc| {
            doc.iter()
                .map(|&value| match mode {
                    ParentCodeMode::Nearest => quantizer.quantize(value),
                    ParentCodeMode::Joint => joint_parent_code(quantizer, value),
                })
                .collect()
        })
        .collect()
}

fn sliced_l2_sqr(
    query: &[f32],
    doc_codes: &[u8],
    quantizer: &MatryoshkaQuantizer,
    bits: u8,
) -> f32 {
    query
        .iter()
        .zip(doc_codes)
        .map(|(&q, &code)| {
            let sliced = quantizer.slice(code, bits);
            let x = quantizer.dequantize(sliced, bits);
            (q - x).powi(2)
        })
        .sum()
}

fn joint_parent_code(quantizer: &MatryoshkaQuantizer, value: f32) -> u8 {
    const PRECISIONS: [u8; 3] = [8, 4, 2];
    const WEIGHTS: [f32; 3] = [0.2, 0.3, 1.0];

    let mut best = (f32::INFINITY, 0u8);
    for code in 0..=u8::MAX {
        let mut weighted_error = 0.0;
        for (&bits, &weight) in PRECISIONS.iter().zip(&WEIGHTS) {
            let sliced = quantizer.slice(code, bits);
            let reconstructed = quantizer.dequantize(sliced, bits);
            weighted_error += weight * (value - reconstructed).powi(2);
        }
        if weighted_error < best.0 {
            best = (weighted_error, code);
        }
    }
    best.1
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn top_k_from_distances(distances: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = distances.iter().copied().enumerate().collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

fn recall_at_k(exact: &[(usize, f32)], approx: &[(usize, f32)]) -> f32 {
    let exact_ids: HashSet<usize> = exact.iter().map(|(idx, _)| *idx).collect();
    let hits = approx
        .iter()
        .filter(|(idx, _)| exact_ids.contains(idx))
        .count();
    hits as f32 / exact.len() as f32
}

fn mean_relative_error(exact: &[f32], approx: &[f32]) -> f32 {
    let mut total = 0.0;
    for (&e, &a) in exact.iter().zip(approx) {
        total += (a - e).abs() / e.max(1e-6);
    }
    total / exact.len() as f32
}

fn make_query(doc: &[f32]) -> Vec<f32> {
    doc.iter()
        .enumerate()
        .map(|(dim, &x)| x + 0.02 * ((dim as f32) * 0.37).sin())
        .collect()
}

fn make_docs() -> Vec<Vec<f32>> {
    (0..N_DOCS)
        .map(|doc_id| {
            let cluster = (doc_id % 8) as f32;
            (0..DIM)
                .map(|dim| {
                    let d = dim as f32;
                    let base = (0.11 * d + cluster).sin() + 0.5 * (0.07 * d - cluster).cos();
                    let local = 0.03 * ((doc_id as f32 + 1.0) * (d + 3.0)).sin();
                    base + local
                })
                .collect()
        })
        .collect()
}
