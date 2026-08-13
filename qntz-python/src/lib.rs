use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// helpers: extract Vec<u8> / Vec<f32> from numpy arrays or lists
// ---------------------------------------------------------------------------

/// Extract `Vec<u8>` from a numpy uint8 array or a Python list of ints.
fn extract_u8_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(arr) = obj.cast::<PyArray1<u8>>() {
        let ro = arr.readonly();
        return Ok(ro.as_slice()?.to_vec());
    }
    obj.extract::<Vec<u8>>()
}

/// Extract `Vec<f32>` from a numpy float32 1D array or a Python list of floats.
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        return Ok(ro.as_slice()?.to_vec());
    }
    obj.extract::<Vec<f32>>()
}

/// Extract flat `Vec<f32>` + `num_vectors` from either:
///   - a 2D numpy array (shape [n, d]) -- num_vectors inferred
///   - a 1D numpy array or flat list + explicit num_vectors
fn extract_training_data(
    vectors_obj: &Bound<'_, PyAny>,
    num_vectors: Option<usize>,
) -> PyResult<(Vec<f32>, usize)> {
    // Try 2D numpy array first
    if let Ok(arr) = vectors_obj.cast::<PyArray2<f32>>() {
        let ro: PyReadonlyArray2<'_, f32> = arr.readonly();
        let shape = ro.shape();
        let n = shape[0];
        let flat: Vec<f32> = ro.as_slice()?.to_vec();
        if let Some(nv) = num_vectors {
            if nv != n {
                return Err(PyValueError::new_err(format!(
                    "num_vectors={nv} does not match array shape[0]={n}"
                )));
            }
        }
        return Ok((flat, n));
    }
    // Fall back to flat extraction
    let flat = extract_f32_vec(vectors_obj)?;
    let nv = num_vectors.ok_or_else(|| {
        PyValueError::new_err("num_vectors is required when vectors is not a 2D numpy array")
    })?;
    Ok((flat, nv))
}

// ---------------------------------------------------------------------------
// simd_ops
// ---------------------------------------------------------------------------

/// Pack binary codes (0/1 values) into a bitfield.
///
/// Accepts a list of ints or a numpy uint8 array. Each value is treated as
/// boolean: nonzero becomes a set bit.
///
/// Returns the packed bytes as a numpy uint8 array.
#[pyfunction]
fn pack_binary<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let codes = extract_u8_vec(codes)?;
    let required = codes.len().div_ceil(8);
    let mut packed = vec![0u8; required];
    qntz_core::simd_ops::pack_binary_fast(&codes, &mut packed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, packed))
}

/// Unpack a bitfield back into one byte per bit (0 or 1).
///
/// Returns a numpy uint8 array of length `dim`.
#[pyfunction]
fn unpack_binary<'py>(
    py: Python<'py>,
    packed: &Bound<'py, PyAny>,
    dim: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let packed = extract_u8_vec(packed)?;
    let mut codes = vec![0u8; dim];
    qntz_core::simd_ops::unpack_binary_fast(&packed, &mut codes, dim)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, codes))
}

/// Hamming distance between two packed bit-vectors.
///
/// Accepts lists of ints, bytes, or numpy uint8 arrays.
#[pyfunction]
fn hamming_distance(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u32> {
    let a = extract_u8_vec(a)?;
    let b = extract_u8_vec(b)?;
    Ok(qntz_core::simd_ops::hamming_distance(&a, &b))
}

/// Asymmetric inner product: f32 query vs packed 1-bit codes.
///
/// Convention: bit=1 -> +1, bit=0 -> -1.
///
/// `query` accepts a list of floats or a numpy float32 array.
/// `codes` accepts a list of ints, bytes, or a numpy uint8 array.
#[pyfunction]
fn asymmetric_inner_product(query: &Bound<'_, PyAny>, codes: &Bound<'_, PyAny>) -> PyResult<f32> {
    let query = extract_f32_vec(query)?;
    let codes = extract_u8_vec(codes)?;
    qntz_core::simd_ops::asymmetric_inner_product(&query, &codes)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Asymmetric L2 distance squared: ||q - b||^2 where b in {-1,+1}^D.
///
/// `query` accepts a list of floats or a numpy float32 array.
/// `codes` accepts a list of ints, bytes, or a numpy uint8 array.
#[pyfunction]
fn asymmetric_l2_squared(query: &Bound<'_, PyAny>, codes: &Bound<'_, PyAny>) -> PyResult<f32> {
    let query = extract_f32_vec(query)?;
    let codes = extract_u8_vec(codes)?;
    qntz_core::simd_ops::asymmetric_l2_squared(&query, &codes)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Multi-bit inner product with centered codes.
#[pyfunction]
fn multibit_inner_product(
    query: &Bound<'_, PyAny>,
    codes: Vec<u16>,
    total_bits: usize,
) -> PyResult<f32> {
    let query = extract_f32_vec(query)?;
    Ok(qntz_core::simd_ops::multibit_inner_product(
        &query, &codes, total_bits,
    ))
}

/// Batch Hamming distances from a query to each element in a list of codes.
///
/// Returns a numpy uint32 array of distances.
#[pyfunction]
fn batch_hamming_distances<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    codes: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let query = extract_u8_vec(query)?;
    let code_vecs: Vec<Vec<u8>> = codes
        .iter()
        .map(|c| extract_u8_vec(c))
        .collect::<PyResult<_>>()?;
    let code_refs: Vec<&[u8]> = code_vecs.iter().map(|v| v.as_slice()).collect();
    let distances = qntz_core::simd_ops::batch_hamming_distances(&query, &code_refs);
    Ok(PyArray1::from_vec(py, distances))
}

/// Batch asymmetric L2 squared distances from a query to each element in a list of codes.
///
/// Returns a numpy float32 array of distances.
#[pyfunction]
fn batch_asymmetric_l2<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    codes: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query = extract_f32_vec(query)?;
    let code_vecs: Vec<Vec<u8>> = codes
        .iter()
        .map(|c| extract_u8_vec(c))
        .collect::<PyResult<_>>()?;
    let code_refs: Vec<&[u8]> = code_vecs.iter().map(|v| v.as_slice()).collect();
    let distances = qntz_core::simd_ops::batch_asymmetric_l2(&query, &code_refs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, distances))
}

// ---------------------------------------------------------------------------
// rabitq
// ---------------------------------------------------------------------------

/// Quantized vector produced by RaBitQ.
///
/// Contains binary codes, extended codes, and metadata needed for
/// approximate distance computation.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct QuantizedVector {
    inner: qntz_core::rabitq::QuantizedVector,
}

#[pymethods]
impl QuantizedVector {
    /// Vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Number of extended bits (total_bits - 1).
    #[getter]
    fn ex_bits(&self) -> u8 {
        self.inner.ex_bits
    }

    /// Quantization step size.
    #[getter]
    fn delta(&self) -> f32 {
        self.inner.delta
    }

    /// Lower bound of quantization range.
    #[getter]
    fn vl(&self) -> f32 {
        self.inner.vl
    }

    /// Additive factor for distance estimation.
    #[getter]
    fn f_add(&self) -> f32 {
        self.inner.f_add
    }

    /// Rescaling factor for distance estimation.
    #[getter]
    fn f_rescale(&self) -> f32 {
        self.inner.f_rescale
    }

    /// Squared-distance error coefficient for a unit-norm query residual.
    #[getter]
    fn f_error(&self) -> f32 {
        self.inner.f_error
    }

    /// Norm of the residual after centroid subtraction.
    #[getter]
    fn residual_norm(&self) -> f32 {
        self.inner.residual_norm
    }

    /// Packed binary codes (1 bit per dimension).
    #[getter]
    fn binary_codes(&self) -> Vec<u8> {
        self.inner.binary_codes.clone()
    }

    /// Extended precision codes.
    #[getter]
    fn extended_codes(&self) -> Vec<u8> {
        self.inner.extended_codes.clone()
    }

    /// Combined multi-bit codes (one u16 per dimension).
    #[getter]
    fn codes(&self) -> Vec<u16> {
        self.inner.codes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizedVector(dimension={}, ex_bits={}, residual_norm={:.4})",
            self.inner.dimension, self.inner.ex_bits, self.inner.residual_norm
        )
    }
}

/// RaBitQ quantizer with extended bit support.
///
/// Quantizes vectors into 1-8 bits per dimension using a random orthogonal
/// rotation and uniform scalar quantization.
///
/// Args:
///     dimension: vector dimension (must be > 0).
///     seed: random seed for rotation matrix.
///     total_bits: bits per dimension, 1-8 (default 4).
#[pyclass]
struct RaBitQQuantizer {
    inner: qntz_core::rabitq::RaBitQQuantizer,
    dimension: usize,
    total_bits: usize,
}

#[pymethods]
impl RaBitQQuantizer {
    #[new]
    #[pyo3(signature = (dimension, seed, total_bits=4))]
    fn new(dimension: usize, seed: u64, total_bits: usize) -> PyResult<Self> {
        let config = qntz_core::rabitq::RaBitQConfig {
            total_bits,
            t_const: None,
        };
        let inner = qntz_core::rabitq::RaBitQQuantizer::with_config(dimension, seed, config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            dimension,
            total_bits,
        })
    }

    /// Fit centroid from training vectors.
    ///
    /// Accepts either:
    ///   - A 2D numpy float32 array of shape (n, dimension) -- num_vectors
    ///     is inferred from shape[0].
    ///   - A flat list or 1D numpy array of floats (row-major) with an
    ///     explicit `num_vectors` parameter.
    #[pyo3(signature = (vectors, num_vectors=None))]
    fn fit(&mut self, vectors: &Bound<'_, PyAny>, num_vectors: Option<usize>) -> PyResult<()> {
        let (flat, nv) = extract_training_data(vectors, num_vectors)?;
        self.inner
            .fit(&flat, nv)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set centroid directly.
    fn set_centroid(&mut self, centroid: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_centroid(centroid)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a single vector.
    ///
    /// Accepts a list of floats or a numpy float32 1D array.
    fn quantize(&self, vector: &Bound<'_, PyAny>) -> PyResult<QuantizedVector> {
        let vector = extract_f32_vec(vector)?;
        let qv = self
            .inner
            .quantize(&vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(QuantizedVector { inner: qv })
    }

    /// Approximate L2 distance squared between query and quantized vector.
    ///
    /// `query` accepts a list of floats or a numpy float32 array.
    fn approximate_l2_sqr(
        &self,
        query: &Bound<'_, PyAny>,
        quantized: &QuantizedVector,
    ) -> PyResult<f32> {
        let query = extract_f32_vec(query)?;
        self.inner
            .approximate_l2_sqr(&query, &quantized.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Query-scaled squared-distance error margin.
    fn squared_distance_error_margin(
        &self,
        query: &Bound<'_, PyAny>,
        quantized: &QuantizedVector,
    ) -> PyResult<f32> {
        let query = extract_f32_vec(query)?;
        self.inner
            .squared_distance_error_margin(&query, &quantized.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Approximate Euclidean distance (sqrt of L2 squared).
    ///
    /// `query` accepts a list of floats or a numpy float32 array.
    fn approximate_distance(
        &self,
        query: &Bound<'_, PyAny>,
        quantized: &QuantizedVector,
    ) -> PyResult<f32> {
        let query = extract_f32_vec(query)?;
        self.inner
            .approximate_distance(&query, &quantized.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RaBitQQuantizer(dimension={}, total_bits={})",
            self.dimension, self.total_bits
        )
    }
}

// ---------------------------------------------------------------------------
// ternary
// ---------------------------------------------------------------------------

/// Ternary quantized vector ({-1, 0, +1} per dimension).
///
/// Stores values in a compact 2-bit encoding. Access individual values
/// with `get()` or retrieve all values with `to_list()`.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct TernaryVector {
    inner: qntz_core::ternary::TernaryVector,
}

#[pymethods]
impl TernaryVector {
    /// Number of dimensions.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// L2 norm of the original (pre-quantization) vector.
    #[getter]
    fn original_norm(&self) -> f32 {
        self.inner.original_norm()
    }

    /// Fraction of dimensions that are zero.
    #[getter]
    fn sparsity(&self) -> f32 {
        self.inner.sparsity()
    }

    /// Memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Get the ternary value at index (returns -1, 0, or +1).
    fn get(&self, idx: usize) -> PyResult<i8> {
        if idx >= self.inner.dimension() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for dimension {}",
                idx,
                self.inner.dimension()
            )));
        }
        Ok(self.inner.get(idx))
    }

    /// Return all ternary values as a list.
    fn to_list(&self) -> Vec<i8> {
        (0..self.inner.dimension())
            .map(|i| self.inner.get(i))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TernaryVector(dimension={}, sparsity={:.2})",
            self.inner.dimension(),
            self.inner.sparsity()
        )
    }
}

/// Ternary quantizer: maps each dimension to {-1, 0, +1}.
///
/// Values above `threshold_high` become +1, below `threshold_low` become -1,
/// and values in between become 0. Optionally L2-normalizes input first.
///
/// Args:
///     dimension: vector dimension.
///     threshold_high: values above this become +1 (default 0.3).
///     threshold_low: values below this become -1 (default -0.3).
///     normalize: L2-normalize before thresholding (default True).
///     target_sparsity: adaptive sparsity target, or None (default None).
#[pyclass]
struct TernaryQuantizer {
    inner: qntz_core::ternary::TernaryQuantizer,
    dimension: usize,
}

#[pymethods]
impl TernaryQuantizer {
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
            dimension,
        }
    }

    /// Fit adaptive thresholds from training vectors.
    ///
    /// Accepts either:
    ///   - A 2D numpy float32 array of shape (n, dimension) -- num_vectors
    ///     is inferred from shape[0].
    ///   - A flat list or 1D numpy array of floats (row-major) with an
    ///     explicit `num_vectors` parameter.
    #[pyo3(signature = (vectors, num_vectors=None))]
    fn fit(&mut self, vectors: &Bound<'_, PyAny>, num_vectors: Option<usize>) -> PyResult<()> {
        let (flat, nv) = extract_training_data(vectors, num_vectors)?;
        self.inner
            .fit(&flat, nv)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Quantize a single vector.
    ///
    /// Accepts a list of floats or a numpy float32 1D array.
    fn quantize(&self, vector: &Bound<'_, PyAny>) -> PyResult<TernaryVector> {
        let vector = extract_f32_vec(vector)?;
        let tv = self
            .inner
            .quantize(&vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TernaryVector { inner: tv })
    }

    fn __repr__(&self) -> String {
        format!("TernaryQuantizer(dimension={})", self.dimension)
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
///
/// `query` accepts a list of floats or a numpy float32 array.
#[pyfunction]
fn ternary_asymmetric_inner_product(
    query: &Bound<'_, PyAny>,
    quantized: &TernaryVector,
) -> PyResult<f32> {
    let query = extract_f32_vec(query)?;
    Ok(qntz_core::ternary::asymmetric_inner_product(
        &query,
        &quantized.inner,
    ))
}

/// Asymmetric cosine distance: 1 - cos(query, quantized).
///
/// `query` accepts a list of floats or a numpy float32 array.
#[pyfunction]
fn ternary_asymmetric_cosine_distance(
    query: &Bound<'_, PyAny>,
    quantized: &TernaryVector,
) -> PyResult<f32> {
    let query = extract_f32_vec(query)?;
    Ok(qntz_core::ternary::asymmetric_cosine_distance(
        &query,
        &quantized.inner,
    ))
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
    m.add("__version__", "0.1.3")?;

    // simd_ops
    m.add_function(wrap_pyfunction!(pack_binary, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_binary, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_l2_squared, m)?)?;
    m.add_function(wrap_pyfunction!(multibit_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hamming_distances, m)?)?;
    m.add_function(wrap_pyfunction!(batch_asymmetric_l2, m)?)?;

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
