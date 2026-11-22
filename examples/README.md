# qntz examples

The examples are deterministic and print quantitative checks rather than timing
claims. Use the benchmarks for throughput measurements.

With no arguments, `sift_rabitq_recall` uses the gitignored SIFT-small dataset fetched by
`scripts/fetch_siftsmall.sh`. If the data is absent, it prints the fetch command
and exits successfully.

## `rabitq_error_budget`

Compares RaBitQ bit widths using observed squared-distance error, the
query-scaled error margin, empirical coverage, and packed code size.

```sh
cargo run --release --features rabitq --example rabitq_error_budget
```

Output:

```text
dataset: 512 docs, dim=64
bits  mean abs squared error  mean error margin  covered  bytes/code
   1                  5.8722            15.2647    94.5%         8.0
   2                  2.6036             7.7498    98.0%        16.0
   4                  0.7975             2.4264    97.5%        32.0
   8                  0.0420             0.1131    95.9%        64.0
```

## `rabitq_statistical_eval`

Measures the unclamped one-bit estimator across random and adversarial vector
families. It reports signed bias, RMSE, tail errors, error-margin coverage, and
rotation orthogonality. The default is a bounded smoke run; `--full` samples
more dimensions and seeds. This is an on-demand evaluation, not a CI test.

```sh
cargo run --release --features rabitq --example rabitq_statistical_eval
cargo run --release --features rabitq --example rabitq_statistical_eval -- --full
```

## `adaptive_scan`

Quantizes a batch with per-vector scalar ranges, scans through a reusable
scan plan, and compares asymmetric distances against exact L2.

```sh
cargo run --release --features adaptive --example adaptive_scan
```

Output:

```text
dataset: 256 docs, dim=96, top-10
bits  recall@10  mean distance relative error  stored code bytes/doc
   2     0.500                        0.4243         96
   4     1.000                        0.0238         96
   8     1.000                        0.0009         96
```

## `matryoshka_precision_scan`

Compares nearest 8-bit parent codes with a joint parent-code objective, then
slices the same stored scalar codes to 2, 4, and 8 bits during scan.

```sh
cargo run --release --features matryoshka --example matryoshka_precision_scan
```

Output:

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

## `additive_codebook_refinement`

Encodes each scalar with ordered additive codebook choices, then scans with the
first 1, 2, or 3 refinement stages active.

```sh
cargo run --release --features matryoshka --example additive_codebook_refinement
```

Output:

```text
dataset: 256 docs, dim=64, top-10
ordered additive codebooks, enabling more stages at scan time
stages  recall@10  mean distance relative error  active bytes/doc
     1     0.400                        0.4848                64
     2     0.500                        0.0807               128
     3     0.300                        0.0126               192
```

## `entropy_coded_quantization`

Entropy-codes RaBitQ scalar codes with ANS and verifies round-trip decoding.

```sh
cargo run --release --features rabitq --example entropy_coded_quantization
```

Output excerpt:

```text
Roundtrip verified: all 32000 symbols match.

Size comparison:
  raw codes (u16):           64000 bytes
  fixed-width (4-bit):       16000 bytes
  ANS entropy-coded:         15433 bytes  (1.04x vs fixed-width)
  theoretical minimum:       15430 bytes  (Shannon entropy)
  ANS overhead vs theory: 0.0%
  bits/symbol: fixed=4, ANS=3.858, entropy=3.857
```

## `sift_rabitq_recall`

Quantizes SIFT-small base vectors with RaBitQ, filters by approximate distance,
reranks the candidate set by exact L2, and reports recall@10.

```sh
./scripts/fetch_siftsmall.sh
cargo run --release --features rabitq --example sift_rabitq_recall
```

Or compare your own base and query vectors:

```sh
cargo run --release --features rabitq --example sift_rabitq_recall -- base.fvecs queries.fvecs
```

Each file contains rows with a little-endian signed 32-bit dimension followed by
that many little-endian float32 coordinates. Both files must have at least one
row and the same positive dimension throughout. Use held-out queries to measure
retrieval on new inputs. The example scans the entire base for exact neighbors.

The table reports packed code payload bytes and the size of one serialized
record, including its correction factors and header. The shared quantizer and
the original vectors used for exact reranking cost additional space. For small
files, k and the candidate budgets are capped at the number of base rows.
