# qntz

Vector quantization primitives for ANN systems.

Scope: small, reusable pieces (bit packing, low-bit codes) that higher-level ANN
crates can build on. This crate intentionally does **not** implement full indices.

## Quickstart

```toml
[dependencies]
qntz = "0.1.0"
```

```rust
use qntz::simd_ops::{hamming_distance, pack_binary_fast};

// Two 8-bit binary vectors (as 0/1 bytes).
let a_bits = [1u8, 0, 1, 0, 1, 0, 1, 0];
let b_bits = [1u8, 1, 1, 0, 0, 0, 1, 0];

let mut a_packed = [0u8; 1];
let mut b_packed = [0u8; 1];
pack_binary_fast(&a_bits, &mut a_packed).unwrap();
pack_binary_fast(&b_bits, &mut b_packed).unwrap();

let d = hamming_distance(&a_packed, &b_packed);
assert_eq!(d, 2);
```

## Features

All quantization modules are feature-gated. The default build provides only the
core bit-packing helpers in `simd_ops`.

| Feature   | What it adds |
|-----------|--------------|
| `rabitq`  | RaBitQ quantizer -- 1-bit sign codes with optional extended bits (up to 8-bit total) and correction factors for approximate L2 distance |
| `ternary` | Ternary (1.58-bit) quantizer -- maps each dimension to {-1, 0, +1} with configurable thresholds and adaptive sparsity |

Enable one or both:

```toml
[dependencies]
qntz = { version = "0.1.0", features = ["rabitq", "ternary"] }
```

## API overview

Fallible operations return `qntz::Result<T>` (wrapping `VQuantError`) for
dimension mismatches and invalid configs. Pure distance helpers that take
pre-validated inputs return scalar values directly.

### `simd_ops` (always available)

Bit packing, Hamming distance, asymmetric inner product / L2, and multi-bit
code operations. All pack/unpack functions return `Result`:

```rust
use qntz::simd_ops::{pack_binary_fast, unpack_binary_fast, hamming_distance};

let codes = [1u8, 0, 1, 1, 0, 0, 1, 0];
let mut packed = [0u8; 1];
pack_binary_fast(&codes, &mut packed).unwrap();

let mut roundtrip = [0u8; 8];
unpack_binary_fast(&packed, &mut roundtrip, 8).unwrap();
assert_eq!(&codes[..], &roundtrip[..]);

let other = [0u8; 1]; // all-zero packed vector
assert_eq!(hamming_distance(&packed, &other), 4); // 4 bits differ
```

### `rabitq` (feature = "rabitq")

RaBitQ quantizer with configurable bit depth (1--8 bits per dimension).
Supports centroid fitting and approximate L2 distance:

```rust
#[cfg(feature = "rabitq")]
{
    use qntz::rabitq::{RaBitQQuantizer, RaBitQConfig};

    let dim = 32;
    let quantizer = RaBitQQuantizer::with_config(
        dim,
        42, // seed for random rotation
        RaBitQConfig::bits4(),
    ).unwrap();

    let vector = vec![0.1f32; dim];
    let quantized = quantizer.quantize(&vector).unwrap();

    // Approximate L2 distance from a query
    let query = vec![0.2f32; dim];
    let dist = quantizer.approximate_distance(&query, &quantized).unwrap();
    assert!(dist >= 0.0);
}
```

### `ternary` (feature = "ternary")

Ternary quantization maps each dimension to {-1, 0, +1} using 2 bits per
value. `ternary_hamming` returns `Option<usize>` (None on dimension mismatch):

```rust
#[cfg(feature = "ternary")]
{
    use qntz::ternary::{TernaryQuantizer, TernaryConfig, ternary_hamming};

    let config = TernaryConfig {
        threshold_high: 0.3,
        threshold_low: -0.3,
        normalize: false,
        target_sparsity: None,
    };
    let quantizer = TernaryQuantizer::new(4, config);

    let a = quantizer.quantize(&[0.5, -0.5, 0.0, 0.0]).unwrap();
    let b = quantizer.quantize(&[0.5,  0.5, 0.0, -0.5]).unwrap();

    // Hamming distance counts differing ternary positions
    assert_eq!(ternary_hamming(&a, &b), Some(2));
}
```

## Design

- **No hidden geometry**: this crate provides code operations. It does not
  impose a distance metric on your original vectors.
- **Determinism**: all operations are pure and deterministic (given the same
  seed for rotation matrices).
- **Error discipline**: dimension mismatches and invalid configs surface as
  typed errors, not panics.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.