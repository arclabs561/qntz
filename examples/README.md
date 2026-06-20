# Examples

The examples are deterministic and print quantitative checks rather than timing
claims. Use the benchmarks for throughput measurements.

## `rabitq_error_budget.rs`

Compares RaBitQ bit widths using the quantizer's error proxy and packed code
size.

```sh
cargo run --release --features rabitq --example rabitq_error_budget
```

## `adaptive_scan.rs`

Quantizes a batch with per-vector scalar ranges and compares asymmetric scan
distances against exact L2.

```sh
cargo run --release --features adaptive --example adaptive_scan
```

## `entropy_coded_quantization.rs`

Shows entropy coding over RaBitQ codes.

```sh
cargo run --release --features rabitq --example entropy_coded_quantization
```
