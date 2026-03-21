use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// simd_ops
// ---------------------------------------------------------------------------

/// Pack binary codes (0/1 bytes) into a bitfield.
///
/// Each input byte is treated as a boolean: nonzero becomes a set bit.
#[pyfunction]
fn pack_binary(codes: Vec<u8>) -> PyResult<Vec<u8>> {
    let required = codes.len().div_ceil(8);
    let mut packed = vec![0u8; required];
    qntz_core::simd_ops::pack_binary_fast(&codes, &mut packed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(packed)
}

/// Unpack a bitfield back into one byte per bit (0 or 1).
#[pyfunction]
fn unpack_binary(packed: Vec<u8>, dim: usize) -> PyResult<Vec<u8>> {
    let mut codes = vec![0u8; dim];
    qntz_core::simd_ops::unpack_binary_fast(&packed, &mut codes, dim)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(codes)
}

/// Hamming distance between two packed bit-vectors.
#[pyfunction]
fn hamming_distance(a: Vec<u8>, b: Vec<u8>) -> u32 {
    qntz_core::simd_ops::hamming_distance(&a, &b)
}

/// Asymmetric inner product: f32 query vs packed 1-bit codes.
///
/// Convention: bit=1 -> +1, bit=0 -> -1.
#[pyfunction]
fn asymmetric_inner_product(query: Vec<f32>, codes: Vec<u8>) -> PyResult<f32> {
    qntz_core::simd_ops::asymmetric_inner_product(&query, &codes)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Asymmetric L2 distance squared: ||q - b||^2 where b in {-1,+1}^D.
#[pyfunction]
fn asymmetric_l2_squared(query: Vec<f32>, codes: Vec<u8>) -> PyResult<f32> {
    qntz_core::simd_ops::asymmetric_l2_squared(&query, &codes)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Multi-bit inner product with centered codes.
#[pyfunction]
fn multibit_inner_product(query: Vec<f32>, codes: Vec<u16>, total_bits: usize) -> f32 {
    qntz_core::simd_ops::multibit_inner_product(&query, &codes, total_bits)
}

// ---------------------------------------------------------------------------
// rabitq
// ---------------------------------------------------------------------------

/// Quantized vector produced by RaBitQ.
#[pyclass]
#[derive(Clone)]
struct QuantizedVector {
    inner: qntz_core::rabitq::QuantizedVector,
}

#[pymethods]
impl QuantizedVector {
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    fn ex_bits(&self) -> u8 {
        self.inner.ex_bits
    }

    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta
    }

    #[getter]
    fn vl(&self) -> f32 {
        self.inner.vl
    }

    #[getter]
    fn f_add(&self) -> f32 {
        self.inner.f_add
    }

    #[getter]
    fn f_rescale(&self) -> f32 {
        self.inner.f_rescale
    }

    #[getter]
    fn f_error(&self) -> f32 {
        self.inner.f_error
    }

    #[getter]
    fn residual_norm(&self) -> f32 {
        self.inner.residual_norm
    }

    #[getter]
    fn binary_codes(&self) -> Vec<u8> {
        self.inner.binary_codes.clone()
    }

    #[getter]
    fn extended_codes(&self) -> Vec<u8> {
        self.inner.extended_codes.clone()
    }

    #[getter]
    fn codes(&self) -> Vec<u16> {
        self.inner.codes.clone()
    }
}

/// RaBitQ quantizer with extended bit support.
#[pyclass]
struct RaBitQQuantizer {
    inner: qntz_core::rabitq::RaBitQQuantizer,
}

#[pymethods]
impl RaBitQQuantizer {
    /// Create a new RaBitQ quantizer.
    ///
    /// Args:
    ///     dimension: vector dimension (must be > 0).
    ///     seed: random seed for rotation matrix.
    ///     total_bits: bits per dimension, 1-8 (default 4).
    #[new]
    #[pyo3(signature = (dimension, seed, total_bits=4))]
    fn new(dimension: usize, seed: u64, total_bits: usize) -> PyResult<Self> {
        let config = qntz_core::rabitq::RaBitQConfig {
            total_bits,
            t_const: None,
        };
        let inner = qntz_core::rabitq::RaBitQQuantizer::with_config(dimension, seed, config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit centroid from training vectors (flat f32 array, row-major).
    fn fit(&mut self, vectors: Vec<f32>, num_vectors: usize) -> PyResult<()> {
        self.inner
            .fit(&vectors, num_vectors)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set centroid directly.
    fn set_centroid(&mut self, centroid: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_centroid(centroid)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a single vector.
    fn quantize(&self, vector: Vec<f32>) -> PyResult<QuantizedVector> {
        let qv = self
            .inner
            .quantize(&vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(QuantizedVector { inner: qv })
    }

    /// Approximate L2 distance squared between query and quantized vector.
    fn approximate_l2_sqr(&self, query: Vec<f32>, quantized: &QuantizedVector) -> PyResult<f32> {
        self.inner
            .approximate_l2_sqr(&query, &quantized.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Approximate Euclidean distance (sqrt of L2 squared).
    fn approximate_distance(
        &self,
        query: Vec<f32>,
        quantized: &QuantizedVector,
    ) -> PyResult<f32> {
        self.inner
            .approximate_distance(&query, &quantized.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ternary
// ---------------------------------------------------------------------------

/// Ternary quantized vector ({-1, 0, +1} per dimension).
#[pyclass]
#[derive(Clone)]
struct TernaryVector {
    inner: qntz_core::ternary::TernaryVector,
}

#[pymethods]
impl TernaryVector {
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn original_norm(&self) -> f32 {
        self.inner.original_norm()
    }

    #[getter]
    fn sparsity(&self) -> f32 {
        self.inner.sparsity()
    }

    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get the ternary value at index (returns -1, 0, or +1).
    fn get(&self, idx: usize) -> i8 {
        self.inner.get(idx)
    }

    /// Return all ternary values as a list.
    fn to_list(&self) -> Vec<i8> {
        (0..self.inner.dimension())
            .map(|i| self.inner.get(i))
            .collect()
    }
}

/// Ternary quantizer: maps each dimension to {-1, 0, +1}.
#[pyclass]
struct TernaryQuantizer {
    inner: qntz_core::ternary::TernaryQuantizer,
}

#[pymethods]
impl TernaryQuantizer {
    /// Create a new ternary quantizer.
    ///
    /// Args:
    ///     dimension: vector dimension.
    ///     threshold_high: values above this become +1 (default 0.3).
    ///     threshold_low: values below this become -1 (default -0.3).
    ///     normalize: L2-normalize before thresholding (default true).
    ///     target_sparsity: adaptive sparsity target, or None (default None).
    #[new]
    #[pyo3(signature = (dimension, threshold_high=0.3, threshold_low=-0.3, normalize=true, target_sparsity=None))]
    fn new(
        dimension: usize,
        threshold_high: f32,
        threshold_low: f32,
        normalize: bool,
        target_sparsity: Option<f32>,
    ) -> Self {
        let config = qntz_core::ternary::TernaryConfig {
            threshold_high,
            threshold_low,
            normalize,
            target_sparsity,
        };
        Self {
            inner: qntz_core::ternary::TernaryQuantizer::new(dimension, config),
        }
    }

    /// Fit adaptive thresholds from training vectors (flat f32, row-major).
    fn fit(&mut self, vectors: Vec<f32>, num_vectors: usize) -> PyResult<()> {
        self.inner
            .fit(&vectors, num_vectors)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a single vector.
    fn quantize(&self, vector: Vec<f32>) -> PyResult<TernaryVector> {
        let tv = self
            .inner
            .quantize(&vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TernaryVector { inner: tv })
    }
}

/// Inner product between two ternary vectors.
#[pyfunction]
fn ternary_inner_product(a: &TernaryVector, b: &TernaryVector) -> i32 {
    qntz_core::ternary::ternary_inner_product(&a.inner, &b.inner)
}

/// Cosine similarity between two ternary vectors.
#[pyfunction]
fn ternary_cosine_similarity(a: &TernaryVector, b: &TernaryVector) -> f32 {
    qntz_core::ternary::ternary_cosine_similarity(&a.inner, &b.inner)
}

/// Asymmetric inner product: f32 query vs ternary codes.
#[pyfunction]
fn ternary_asymmetric_inner_product(query: Vec<f32>, quantized: &TernaryVector) -> f32 {
    qntz_core::ternary::asymmetric_inner_product(&query, &quantized.inner)
}

/// Asymmetric cosine distance: 1 - cos(query, quantized).
#[pyfunction]
fn ternary_asymmetric_cosine_distance(query: Vec<f32>, quantized: &TernaryVector) -> f32 {
    qntz_core::ternary::asymmetric_cosine_distance(&query, &quantized.inner)
}

/// Hamming distance between two ternary vectors.
///
/// Returns None if dimensions differ.
#[pyfunction]
fn ternary_hamming(a: &TernaryVector, b: &TernaryVector) -> Option<usize> {
    qntz_core::ternary::ternary_hamming(&a.inner, &b.inner)
}

// ---------------------------------------------------------------------------
// module
// ---------------------------------------------------------------------------

#[pymodule]
fn qntz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // simd_ops
    m.add_function(wrap_pyfunction!(pack_binary, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_binary, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_l2_squared, m)?)?;
    m.add_function(wrap_pyfunction!(multibit_inner_product, m)?)?;

    // rabitq
    m.add_class::<QuantizedVector>()?;
    m.add_class::<RaBitQQuantizer>()?;

    // ternary
    m.add_class::<TernaryVector>()?;
    m.add_class::<TernaryQuantizer>()?;
    m.add_function(wrap_pyfunction!(ternary_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_asymmetric_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_asymmetric_cosine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_hamming, m)?)?;

    Ok(())
}
