# qntz

Vector quantization primitives for approximate nearest neighbor search: RaBitQ (1-8 bit), ternary quantization, bit packing, and Hamming distance. Backed by a Rust implementation.

Python bindings for the [qntz](https://crates.io/crates/qntz) Rust crate.

## Install

    pip install qntz

## Quick start

```python
import qntz

# RaBitQ quantization (1-8 bits per dimension)
dim = 32
quantizer = qntz.RaBitQQuantizer(dim, seed=42, total_bits=4)
vector = [float(i) * 0.1 for i in range(dim)]
qv = quantizer.quantize(vector)
query = [float(i) * 0.05 for i in range(dim)]
dist = quantizer.approximate_distance(query, qv)

# Ternary quantization ({-1, 0, +1} per dimension)
tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
a = tq.quantize([0.5, -0.5, 0.1, -0.1])
b = tq.quantize([0.5, 0.5, 0.0, -0.5])
sim = qntz.ternary_cosine_similarity(a, b)

# Bit packing and Hamming distance
packed_a = qntz.pack_binary([1, 0, 1, 0, 1, 0, 1, 0])
packed_b = qntz.pack_binary([1, 1, 1, 0, 0, 0, 1, 0])
dist = qntz.hamming_distance(packed_a, packed_b)  # 2
```

## API

| Name | Description |
|------|-------------|
| `RaBitQQuantizer` | Quantizer using randomized Hadamard rotation, 1-8 bits/dim |
| `RaBitQQuantizer.fit` | Fit centroid from training vectors |
| `RaBitQQuantizer.quantize` | Quantize a single vector, returns `QuantizedVector` |
| `RaBitQQuantizer.approximate_distance` | Approximate Euclidean distance to a quantized vector |
| `RaBitQQuantizer.approximate_l2_sqr` | Approximate squared L2 distance to a quantized vector |
| `TernaryQuantizer` | Maps each dimension to {-1, 0, +1} via thresholds |
| `TernaryQuantizer.fit` | Fit adaptive thresholds from training vectors |
| `TernaryQuantizer.quantize` | Quantize a single vector, returns `TernaryVector` |
| `pack_binary` | Pack 0/1 values into a compact bitfield |
| `unpack_binary` | Unpack a bitfield back to one byte per bit |
| `hamming_distance` | Hamming distance between two packed bit-vectors |
| `asymmetric_inner_product` | f32 query vs packed 1-bit codes inner product |
| `asymmetric_l2_squared` | f32 query vs packed 1-bit codes L2 squared |
| `ternary_inner_product` | Inner product between two ternary vectors |
| `ternary_cosine_similarity` | Cosine similarity between two ternary vectors |
| `ternary_hamming` | Hamming distance between two ternary vectors |
| `ternary_asymmetric_inner_product` | f32 query vs ternary codes inner product |
| `ternary_asymmetric_cosine_distance` | f32 query vs ternary codes cosine distance |

## numpy support

All functions accept numpy arrays or Python lists. Packed outputs are numpy uint8 arrays.

## License

MIT OR Apache-2.0
