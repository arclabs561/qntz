# Examples

The examples are deterministic and print quantitative checks rather than timing
claims. Use the benchmarks for throughput measurements.

## `rabitq_error_budget.rs`

Compares RaBitQ bit widths using the quantizer's error proxy and packed code
size.

```sh
cargo run --release --features rabitq --example rabitq_error_budget
```

Expected output:

```text
dataset: 512 docs, dim=64
bits  mean relative error proxy  mean residual norm  bytes/code
   1                     0.0561              6.3110         8.0
   2                     0.0286              6.3110        16.0
   4                     0.0091              6.3110        32.0
   8                     0.0004              6.3110        64.0
```

## `adaptive_scan.rs`

Quantizes a batch with per-vector scalar ranges and compares asymmetric scan
distances against exact L2.

```sh
cargo run --release --features adaptive --example adaptive_scan
```

Expected output:

```text
dataset: 256 docs, dim=96, top-10
bits  recall@10  mean distance relative error  stored code bytes/doc
   2     0.500                        0.4243         96
   4     1.000                        0.0238         96
   8     1.000                        0.0009         96
```

## `entropy_coded_quantization.rs`

Shows entropy coding over RaBitQ codes.

```sh
cargo run --release --features rabitq --example entropy_coded_quantization
```

Expected excerpt:

```text
Roundtrip verified: all 32000 symbols match.

Size comparison:
  raw codes (u16):           64000 bytes
  fixed-width (4-bit):       16000 bytes
  ANS entropy-coded:         15433 bytes  (1.04x vs fixed-width)
  theoretical minimum:       15430 bytes  (Shannon entropy)
  bits/symbol: fixed=4, ANS=3.858, entropy=3.857
```
