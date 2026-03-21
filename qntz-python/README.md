# qntz (Python)

Python bindings for the [qntz](https://crates.io/crates/qntz) Rust crate.

Vector quantization primitives: RaBitQ (1-bit through 8-bit), ternary quantization, bit packing, and Hamming distance.

## Install

```
pip install qntz
```

## Usage

### Bit packing and Hamming distance

```python
import qntz

codes_a = [1, 0, 1, 0, 1, 0, 1, 0]
codes_b = [1, 1, 1, 0, 0, 0, 1, 0]

packed_a = qntz.pack_binary(codes_a)
packed_b = qntz.pack_binary(codes_b)

dist = qntz.hamming_distance(packed_a, packed_b)
# dist == 2
```

### RaBitQ quantization

```python
import qntz

dim = 32
quantizer = qntz.RaBitQQuantizer(dim, seed=42, total_bits=4)

vector = [float(i) * 0.1 for i in range(dim)]
qv = quantizer.quantize(vector)

query = [float(i) * 0.05 for i in range(dim)]
dist = quantizer.approximate_distance(query, qv)
```

### Ternary quantization

```python
import qntz

tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)

a = tq.quantize([0.5, -0.5, 0.1, -0.1])
b = tq.quantize([0.5, 0.5, 0.0, -0.5])

hamming = qntz.ternary_hamming(a, b)
ip = qntz.ternary_inner_product(a, b)
cos = qntz.ternary_cosine_similarity(a, b)
```

## License

MIT OR Apache-2.0
