//! Drop-by-Drop-style additive codebook refinement.
//!
//! This is a tiny scalar analog of additive-codebook multi-bitwidth
//! quantization. Each value is encoded by an ordered list of codebook choices.
//! Reconstruction with the first stage gives a coarse approximation; enabling
//! later stages adds refinement terms from later codebooks.
//!
//! Run: cargo run --release --features matryoshka --example additive_codebook_refinement

use std::collections::HashSet;

const N_DOCS: usize = 256;
const DIM: usize = 64;
const TOP_K: usize = 10;
const STAGES: usize = 3;
const LEVELS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct AdditiveCodebooks {
    levels: [[f32; LEVELS]; STAGES],
}

impl AdditiveCodebooks {
    fn fit(values: &[f32]) -> Self {
        let mut residuals = values.to_vec();
        let mut levels = [[0.0; LEVELS]; STAGES];

        for stage_levels in &mut levels {
            let radius = residuals
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            *stage_levels = [-radius, -radius / 3.0, radius / 3.0, radius];

            for residual in &mut residuals {
                let code = nearest_level(*residual, stage_levels);
                *residual -= stage_levels[code as usize];
            }
        }

        Self { levels }
    }

    fn encode(&self, value: f32) -> [u8; STAGES] {
        let mut residual = value;
        let mut codes = [0u8; STAGES];
        for (stage, stage_levels) in self.levels.iter().enumerate() {
            let code = nearest_level(residual, stage_levels);
            codes[stage] = code;
            residual -= stage_levels[code as usize];
        }
        codes
    }

    fn reconstruct(&self, codes: &[u8; STAGES], active_stages: usize) -> f32 {
        self.levels
            .iter()
            .zip(codes)
            .take(active_stages)
            .map(|(stage_levels, &code)| stage_levels[code as usize])
            .sum()
    }
}

fn main() {
    let docs = make_docs();
    let query = make_query(&docs[37], &docs[92]);
    let values: Vec<f32> = docs.iter().flat_map(|doc| doc.iter().copied()).collect();
    let codebooks = AdditiveCodebooks::fit(&values);
    let codes: Vec<Vec<[u8; STAGES]>> = docs
        .iter()
        .map(|doc| doc.iter().map(|&value| codebooks.encode(value)).collect())
        .collect();

    let exact_distances: Vec<f32> = docs.iter().map(|doc| l2_sqr(&query, doc)).collect();
    let exact_top = top_k_from_distances(&exact_distances, TOP_K);
    let mut rows = Vec::new();

    println!("dataset: {N_DOCS} docs, dim={DIM}, top-{TOP_K}");
    println!("ordered additive codebooks, enabling more stages at scan time");
    println!("stages  recall@10  mean distance relative error  active bytes/doc");

    for active_stages in 1..=STAGES {
        let approx_distances: Vec<f32> = codes
            .iter()
            .map(|doc_codes| sliced_l2_sqr(&query, doc_codes, &codebooks, active_stages))
            .collect();
        let approx_top = top_k_from_distances(&approx_distances, TOP_K);
        let recall = recall_at_k(&exact_top, &approx_top);
        let mean_rel_err = mean_relative_error(&exact_distances, &approx_distances);
        let active_bytes = active_stages * DIM;

        println!("{active_stages:>6}  {recall:>8.3}  {mean_rel_err:>28.4}  {active_bytes:>16}");
        rows.push((active_stages, recall, mean_rel_err));
    }

    let err1 = rows.iter().find(|(stage, _, _)| *stage == 1).unwrap().2;
    let err2 = rows.iter().find(|(stage, _, _)| *stage == 2).unwrap().2;
    let err3 = rows.iter().find(|(stage, _, _)| *stage == 3).unwrap().2;

    assert!(err3 < err2 && err2 < err1);
    assert!(rows.iter().all(|(_, recall, _)| *recall > 0.0));
}

fn nearest_level(value: f32, levels: &[f32; LEVELS]) -> u8 {
    levels
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (value - **a).abs();
            let db = (value - **b).abs();
            da.total_cmp(&db)
        })
        .map(|(index, _)| index as u8)
        .unwrap()
}

fn sliced_l2_sqr(
    query: &[f32],
    doc_codes: &[[u8; STAGES]],
    codebooks: &AdditiveCodebooks,
    active_stages: usize,
) -> f32 {
    query
        .iter()
        .zip(doc_codes)
        .map(|(&q, codes)| {
            let x = codebooks.reconstruct(codes, active_stages);
            (q - x).powi(2)
        })
        .sum()
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

fn make_query(first: &[f32], second: &[f32]) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .enumerate()
        .map(|(dim, (&a, &b))| 0.85 * a + 0.15 * b + 0.02 * ((dim as f32) * 0.37).sin())
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
