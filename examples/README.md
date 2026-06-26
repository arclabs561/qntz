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

## `matryoshka_precision_scan.rs`

Compares nearest 8-bit parent codes with a joint parent-code objective, then
slices the same stored scalar codes to 2, 4, and 8 bits during scan.

```sh
cargo run --release --features matryoshka --example matryoshka_precision_scan
```

Expected output:

```text
dataset: 256 docs, dim=64, top-10
one stored 8-bit code per scalar, sliced at query time
mode     bits  recall@10  mean distance relative error  stored bytes/doc
nearest     2     0.600                        6.0608                64
nearest     4     0.500                        0.4809                64
nearest     8     0.900                        0.0035                64
  joint     2     0.400                        6.0471                64
  joint     4     0.400                        0.4694                64
  joint     8     0.800                        0.0040                64
```

## `additive_codebook_refinement.rs`

Encodes each scalar with ordered additive codebook choices, then scans with the
first 1, 2, or 3 refinement stages active.

```sh
cargo run --release --features matryoshka --example additive_codebook_refinement
```

Expected output:

```text
dataset: 256 docs, dim=64, top-10
ordered additive codebooks, enabling more stages at scan time
stages  recall@10  mean distance relative error  active bytes/doc
     1     0.400                        0.4848                64
     2     0.500                        0.0807               128
     3     0.300                        0.0126               192
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
