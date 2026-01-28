# qntz

Vector quantization primitives for ANN systems.

Scope: small, reusable pieces (bit packing, low-bit codes) that higher-level ANN
crates can build on. This crate intentionally does **not** implement full indices.

## Features

- `rabitq`: RaBitQ-style 1-bit codes (when enabled)
- `ternary`: ternary codes (when enabled)

## Example

```rust
use qntz::simd_ops::{hamming_distance, pack_binary_fast};

// Two 8-bit binary vectors (as 0/1 bytes).
let a_bits = [1u8, 0, 1, 0, 1, 0, 1, 0];
let b_bits = [1u8, 1, 1, 0, 0, 0, 1, 0];

let mut a_packed = [0u8; 1];
let mut b_packed = [0u8; 1];
pack_binary_fast(&a_bits, &mut a_packed);
pack_binary_fast(&b_bits, &mut b_packed);

let d = hamming_distance(&a_packed, &b_packed);
assert_eq!(d, 2);
```

