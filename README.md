# qntz

Vector quantization.

```toml
[dependencies]
qntz = "0.2.1"
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

See the runnable [examples](examples/README.md) for the optional quantizers.

qntz provides quantization and code operations, not an ANN index; validation
and error behavior are specific to each API.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
