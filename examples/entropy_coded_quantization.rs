//! Entropy-coded quantization: RaBitQ + ANS.
//!
//! Generates vectors, fits RaBitQ, entropy-codes scalar codes with ANS, then
//! verifies round-trip decoding and reports fixed-width vs entropy-coded size.
//!
//! Run:
//!
//! ```sh
//! cargo run --release --features rabitq --example entropy_coded_quantization
//! ```

use ans::{decode, encode, FrequencyTable};
use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

// Simple xorshift64 PRNG (no extra deps).
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f32 in (-1, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

fn main() {
    let dim = 64;
    let num_vectors = 500;
    let total_bits: usize = 4; // 1 sign + 3 extended -> 16 possible code values
    let seed = 0xCAFE_BABE;

    let mut rng = Xorshift64::new(seed);
    let mut flat_vectors = Vec::with_capacity(num_vectors * dim);
    for _ in 0..num_vectors * dim {
        flat_vectors.push(rng.next_f32());
    }

    let config = RaBitQConfig {
        total_bits,
        t_const: None,
    };
    let mut quantizer = RaBitQQuantizer::with_config(dim, seed, config).unwrap();
    quantizer.fit(&flat_vectors, num_vectors).unwrap();

    let alphabet_size = 1usize << total_bits; // 16 for 4-bit
    let mut all_codes: Vec<u32> = Vec::with_capacity(num_vectors * dim);
    let mut freq_counts = vec![0u32; alphabet_size];

    for i in 0..num_vectors {
        let vec_slice = &flat_vectors[i * dim..(i + 1) * dim];
        let qv = quantizer.quantize(vec_slice).unwrap();
        for &code in &qv.codes {
            let c = code as u32;
            all_codes.push(c);
            freq_counts[c as usize] += 1;
        }
    }

    let total_symbols = all_codes.len();

    println!("Code distribution ({total_symbols} symbols, {alphabet_size} alphabet):");
    let empirical_entropy = {
        let mut h = 0.0f64;
        let n = total_symbols as f64;
        for (sym, &count) in freq_counts.iter().enumerate() {
            let frac = count as f64 / n;
            if frac > 0.0 {
                h -= frac * frac.log2();
                println!("  sym {sym:>2}: count {count:>6}  ({:.1}%)", frac * 100.0);
            }
        }
        h
    };
    println!("  empirical entropy: {empirical_entropy:.3} bits/symbol");

    // We only need entries for symbols that appear. But FrequencyTable::from_counts
    // expects a contiguous alphabet 0..N, and we need to encode symbols by their
    // original value. So we keep the full alphabet and replace zeros with 1 (minimal
    // frequency) to avoid ZeroFrequency errors if any unused symbol is somehow encoded.
    let adjusted_counts: Vec<u32> = freq_counts
        .iter()
        .map(|&c| if c == 0 { 1 } else { c })
        .collect();

    let precision_bits = 14;
    let table = FrequencyTable::from_counts(&adjusted_counts, precision_bits)
        .expect("failed to build frequency table");

    let encoded_bytes = encode(&all_codes, &table).expect("ANS encode failed");

    let decoded = decode(&encoded_bytes, &table, total_symbols).expect("ANS decode failed");
    assert_eq!(
        all_codes, decoded,
        "roundtrip mismatch: ANS decode differs from original codes"
    );
    println!("\nRoundtrip verified: all {total_symbols} symbols match.");

    let raw_bytes = total_symbols * 2; // u16 per code
    let fixed_bits = total_symbols * total_bits;
    let fixed_bytes = fixed_bits.div_ceil(8);
    let ans_bytes = encoded_bytes.len();
    let theoretical_bytes = ((empirical_entropy * total_symbols as f64) / 8.0).ceil() as usize;

    println!("\nSize comparison:");
    println!("  raw codes (u16):        {raw_bytes:>8} bytes");
    println!("  fixed-width ({total_bits}-bit):    {fixed_bytes:>8} bytes");
    println!(
        "  ANS entropy-coded:      {:>8} bytes  ({:.2}x vs fixed-width)",
        ans_bytes,
        fixed_bytes as f64 / ans_bytes as f64
    );
    println!("  theoretical minimum:    {theoretical_bytes:>8} bytes  (Shannon entropy)");
    println!(
        "  ANS overhead vs theory: {:.1}%",
        (ans_bytes as f64 / theoretical_bytes as f64 - 1.0) * 100.0
    );
    println!(
        "  bits/symbol: fixed={total_bits:.1}, ANS={:.3}, entropy={empirical_entropy:.3}",
        (ans_bytes as f64 * 8.0) / total_symbols as f64
    );
}
