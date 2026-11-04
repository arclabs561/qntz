# qntz

Vector quantization.

```toml
[dependencies]
qntz = "0.3"
```

```rust
use qntz::simd_ops::{hamming_distance, pack_binary_fast};

let a = [1u8, 0, 1, 0, 1, 0, 1, 0];
let b = [1u8, 1, 1, 0, 0, 0, 1, 0];
let mut packed_a = [0u8; 1];
let mut packed_b = [0u8; 1];

pack_binary_fast(&a, &mut packed_a)?;
pack_binary_fast(&b, &mut packed_b)?;
assert_eq!(hamming_distance(&packed_a, &packed_b), 2);

# Ok::<(), qntz::VQuantError>(())
```

## Features

| Feature | Provides |
|---|---|
| default | bit packing and distance helpers |
| `simd` | dispatch through `innr` where supported |
| `rabitq` | RaBitQ quantization |
| `ternary` | ternary quantization |
| `distquant` | distribution-aware scalar quantization |
| `adaptive` | per-vector adaptive scalar quantization |
| `binary` | rotation-based binary quantization |
| `matryoshka` | scalar codes scanned at several precisions |
| `serde` | derives on selected binary, RaBitQ, and matryoshka types |

The [examples](examples/README.md) exercise the optional quantizers. For
instance, `rabitq_error_budget` quantizes 512 clustered 64-dimensional vectors
at several RaBitQ bit widths and checks how often the reported error margin
covers the observed squared-distance error:

```sh
cargo run --release --features rabitq --example rabitq_error_budget
```

```text
dataset: 512 docs, dim=64
bits  mean abs squared error  mean error margin  covered  bytes/code
   1                  5.8722            15.2647    94.5%         8.0
   2                  2.6036             7.7498    98.0%        16.0
   4                  0.7975             2.4264    97.5%        32.0
   8                  0.0420             0.1131    95.9%        64.0
```

The Rust library requires Rust 1.80 without `simd`. The optional `simd`
feature requires Rust 1.89 through `innr`.

qntz provides quantization and code operations, not an ANN index; validation
and error behavior are specific to each API.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
