"""Type stubs for qntz -- Python bindings to the qntz Rust crate.

Note: NaN values in float inputs will propagate through quantization silently.
Filter NaN before calling if this is not desired.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

__version__: str

# -- simd_ops --

def pack_binary(codes: Union[list[int], npt.NDArray[np.uint8]]) -> npt.NDArray[np.uint8]:
    """Pack binary codes (0/1 values) into a bitfield.

    Accepts a list of ints or a numpy uint8 array. Each value is treated
    as boolean: nonzero becomes a set bit.

    Returns the packed bytes as a numpy uint8 array.
    """
    ...

def unpack_binary(
    packed: Union[list[int], bytes, npt.NDArray[np.uint8]], dim: int
) -> npt.NDArray[np.uint8]:
    """Unpack a bitfield back into one byte per bit (0 or 1).

    Returns a numpy uint8 array of length ``dim``.
    """
    ...

def hamming_distance(
    a: Union[list[int], bytes, npt.NDArray[np.uint8]],
    b: Union[list[int], bytes, npt.NDArray[np.uint8]],
) -> int:
    """Hamming distance between two packed bit-vectors."""
    ...

def asymmetric_inner_product(
    query: Union[list[float], npt.NDArray[np.float32]],
    codes: Union[list[int], bytes, npt.NDArray[np.uint8]],
) -> float:
    """Asymmetric inner product: f32 query vs packed 1-bit codes.

    Convention: bit=1 -> +1, bit=0 -> -1.
    """
    ...

def asymmetric_l2_squared(
    query: Union[list[float], npt.NDArray[np.float32]],
    codes: Union[list[int], bytes, npt.NDArray[np.uint8]],
) -> float:
    """Asymmetric L2 distance squared: ||q - b||^2 where b in {-1,+1}^D."""
    ...

def multibit_inner_product(
    query: Union[list[float], npt.NDArray[np.float32]], codes: list[int], total_bits: int
) -> float:
    """Multi-bit inner product with centered codes."""
    ...

# -- rabitq --

class QuantizedVector:
    """Quantized vector produced by RaBitQ.

    Contains binary codes, extended codes, and metadata needed for
    approximate distance computation.
    """

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        ...
    @property
    def ex_bits(self) -> int:
        """Number of extended bits (total_bits - 1)."""
        ...
    @property
    def delta(self) -> float:
        """Quantization step size."""
        ...
    @property
    def vl(self) -> float:
        """Lower bound of quantization range."""
        ...
    @property
    def f_add(self) -> float:
        """Additive factor for distance estimation."""
        ...
    @property
    def f_rescale(self) -> float:
        """Rescaling factor for distance estimation."""
        ...
    @property
    def f_error(self) -> float:
        """Quantization error bound."""
        ...
    @property
    def residual_norm(self) -> float:
        """Norm of the residual after centroid subtraction."""
        ...
    @property
    def binary_codes(self) -> list[int]:
        """Packed binary codes (1 bit per dimension)."""
        ...
    @property
    def extended_codes(self) -> list[int]:
        """Extended precision codes."""
        ...
    @property
    def codes(self) -> list[int]:
        """Combined multi-bit codes (one u16 per dimension)."""
        ...
    def __repr__(self) -> str: ...

class RaBitQQuantizer:
    """RaBitQ quantizer with extended bit support.

    Quantizes vectors into 1-8 bits per dimension using randomized
    Hadamard rotation and uniform scalar quantization.

    Args:
        dimension: vector dimension (must be > 0).
        seed: random seed for rotation matrix.
        total_bits: bits per dimension, 1-8 (default 4).
    """

    def __init__(self, dimension: int, seed: int, total_bits: int = 4) -> None: ...
    def fit(
        self,
        vectors: Union[list[float], npt.NDArray[np.float32]],
        num_vectors: Optional[int] = None,
    ) -> None:
        """Fit centroid from training vectors.

        Accepts a 2D numpy float32 array (num_vectors inferred from shape[0])
        or a flat list / 1D array with explicit ``num_vectors``.
        """
        ...
    def set_centroid(self, centroid: list[float]) -> None:
        """Set centroid directly."""
        ...
    def quantize(
        self, vector: Union[list[float], npt.NDArray[np.float32]]
    ) -> QuantizedVector:
        """Quantize a single vector."""
        ...
    def approximate_l2_sqr(
        self,
        query: Union[list[float], npt.NDArray[np.float32]],
        quantized: QuantizedVector,
    ) -> float:
        """Approximate L2 distance squared between query and quantized vector."""
        ...
    def approximate_distance(
        self,
        query: Union[list[float], npt.NDArray[np.float32]],
        quantized: QuantizedVector,
    ) -> float:
        """Approximate Euclidean distance (sqrt of L2 squared)."""
        ...
    def __repr__(self) -> str: ...

# -- ternary --

class TernaryVector:
    """Ternary quantized vector ({-1, 0, +1} per dimension).

    Stores values in a compact 2-bit encoding.
    """

    @property
    def dimension(self) -> int:
        """Number of dimensions."""
        ...
    @property
    def original_norm(self) -> float:
        """L2 norm of the original (pre-quantization) vector."""
        ...
    @property
    def sparsity(self) -> float:
        """Fraction of dimensions that are zero."""
        ...
    @property
    def memory_bytes(self) -> int:
        """Memory usage in bytes."""
        ...
    def get(self, idx: int) -> int:
        """Get the ternary value at index (returns -1, 0, or +1)."""
        ...
    def to_list(self) -> list[int]:
        """Return all ternary values as a list."""
        ...
    def __repr__(self) -> str: ...

class TernaryQuantizer:
    """Ternary quantizer: maps each dimension to {-1, 0, +1}.

    Values above ``threshold_high`` become +1, below ``threshold_low``
    become -1, and values in between become 0.

    Args:
        dimension: vector dimension.
        threshold_high: values above this become +1 (default 0.3).
        threshold_low: values below this become -1 (default -0.3).
        normalize: L2-normalize before thresholding (default True).
        target_sparsity: adaptive sparsity target, or None (default None).
    """

    def __init__(
        self,
        dimension: int,
        threshold_high: float = 0.3,
        threshold_low: float = -0.3,
        normalize: bool = True,
        target_sparsity: Optional[float] = None,
    ) -> None: ...
    def fit(
        self,
        vectors: Union[list[float], npt.NDArray[np.float32]],
        num_vectors: Optional[int] = None,
    ) -> None:
        """Fit adaptive thresholds from training vectors.

        Accepts a 2D numpy float32 array (num_vectors inferred from shape[0])
        or a flat list / 1D array with explicit ``num_vectors``.
        """
        ...
    def quantize(
        self, vector: Union[list[float], npt.NDArray[np.float32]]
    ) -> TernaryVector:
        """Quantize a single vector."""
        ...
    def __repr__(self) -> str: ...

def ternary_inner_product(a: TernaryVector, b: TernaryVector) -> int:
    """Inner product between two ternary vectors."""
    ...

def ternary_cosine_similarity(a: TernaryVector, b: TernaryVector) -> float:
    """Cosine similarity between two ternary vectors."""
    ...

def ternary_asymmetric_inner_product(
    query: Union[list[float], npt.NDArray[np.float32]], quantized: TernaryVector
) -> float:
    """Asymmetric inner product: f32 query vs ternary codes."""
    ...

def ternary_asymmetric_cosine_distance(
    query: Union[list[float], npt.NDArray[np.float32]], quantized: TernaryVector
) -> float:
    """Asymmetric cosine distance: 1 - cos(query, quantized)."""
    ...

def ternary_hamming(a: TernaryVector, b: TernaryVector) -> Optional[int]:
    """Hamming distance between two ternary vectors.

    Returns None if dimensions differ.
    """
    ...
