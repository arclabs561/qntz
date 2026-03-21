"""Tests for qntz Python bindings."""

import math

import numpy as np
import pytest

import qntz


# ---------------------------------------------------------------------------
# pack / unpack binary
# ---------------------------------------------------------------------------


class TestPackBinary:
    def test_roundtrip(self):
        codes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
        packed = qntz.pack_binary(codes)
        assert len(packed) == 2
        unpacked = qntz.unpack_binary(packed, len(codes))
        assert list(unpacked) == codes

    def test_empty(self):
        packed = qntz.pack_binary([])
        assert len(packed) == 0
        unpacked = qntz.unpack_binary(b"", 0)
        assert len(unpacked) == 0

    def test_nonzero_treated_as_one(self):
        codes = [5, 0, 255, 0, 0, 0, 0, 0]
        packed = qntz.pack_binary(codes)
        unpacked = qntz.unpack_binary(packed, 8)
        assert list(unpacked) == [1, 0, 1, 0, 0, 0, 0, 0]

    def test_partial_byte(self):
        codes = [1, 0, 1]
        packed = qntz.pack_binary(codes)
        assert len(packed) == 1
        unpacked = qntz.unpack_binary(packed, 3)
        assert list(unpacked) == [1, 0, 1]

    def test_numpy_input(self):
        codes = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        packed = qntz.pack_binary(codes)
        assert len(packed) == 1

    def test_unpack_returns_numpy(self):
        packed = qntz.pack_binary([1, 0, 1, 1, 0, 0, 1, 0])
        unpacked = qntz.unpack_binary(packed, 8)
        assert isinstance(unpacked, np.ndarray)
        assert unpacked.dtype == np.uint8
        assert list(unpacked) == [1, 0, 1, 1, 0, 0, 1, 0]

    def test_numpy_roundtrip(self):
        codes = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
        packed = qntz.pack_binary(codes)
        unpacked = qntz.unpack_binary(packed, len(codes))
        np.testing.assert_array_equal(unpacked, codes)


# ---------------------------------------------------------------------------
# hamming distance
# ---------------------------------------------------------------------------


class TestHammingDistance:
    def test_identical(self):
        a = [0xAB] * 10
        assert qntz.hamming_distance(a, a) == 0

    def test_opposite(self):
        assert qntz.hamming_distance([0xFF], [0x00]) == 8

    def test_empty(self):
        assert qntz.hamming_distance([], []) == 0

    def test_known(self):
        assert qntz.hamming_distance([0b11110000], [0b00001111]) == 8
        assert qntz.hamming_distance([0b11110000], [0b11110000]) == 0
        assert qntz.hamming_distance([0b11110000], [0b11100000]) == 1

    def test_numpy_input(self):
        a = np.array([0xFF], dtype=np.uint8)
        b = np.array([0x00], dtype=np.uint8)
        assert qntz.hamming_distance(a, b) == 8

    def test_mixed_numpy_list(self):
        a = np.array([0xFF], dtype=np.uint8)
        assert qntz.hamming_distance(a, [0x00]) == 8


# ---------------------------------------------------------------------------
# asymmetric inner product / L2
# ---------------------------------------------------------------------------


class TestAsymmetricOps:
    def test_all_positive(self):
        query = [1.0] * 8
        codes = [0xFF]  # all +1
        ip = qntz.asymmetric_inner_product(query, codes)
        assert abs(ip - 8.0) < 1e-6

    def test_all_negative(self):
        query = [1.0] * 8
        codes = [0x00]  # all -1
        ip = qntz.asymmetric_inner_product(query, codes)
        assert abs(ip - (-8.0)) < 1e-6

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            qntz.asymmetric_inner_product([1.0] * 16, [0xFF])

    def test_l2_nonnegative(self):
        query = [1.0] * 8
        codes = [0b10101010]
        l2 = qntz.asymmetric_l2_squared(query, codes)
        assert l2 >= 0.0

    def test_numpy_query(self):
        query = np.array([1.0] * 8, dtype=np.float32)
        codes = [0xFF]
        ip = qntz.asymmetric_inner_product(query, codes)
        assert abs(ip - 8.0) < 1e-6

    def test_numpy_both(self):
        query = np.array([1.0] * 8, dtype=np.float32)
        codes = np.array([0xFF], dtype=np.uint8)
        ip = qntz.asymmetric_inner_product(query, codes)
        assert abs(ip - 8.0) < 1e-6

    def test_l2_numpy(self):
        query = np.array([1.0] * 8, dtype=np.float32)
        codes = np.array([0b10101010], dtype=np.uint8)
        l2 = qntz.asymmetric_l2_squared(query, codes)
        assert l2 >= 0.0


# ---------------------------------------------------------------------------
# multibit inner product
# ---------------------------------------------------------------------------


class TestMultibitIP:
    def test_centered(self):
        query = [1.0, 1.0, 1.0, 1.0]
        codes = [15, 15, 0, 0]
        ip = qntz.multibit_inner_product(query, codes, 4)
        assert abs(ip) < 1e-6


# ---------------------------------------------------------------------------
# RaBitQ
# ---------------------------------------------------------------------------


class TestRaBitQ:
    def test_binary_quantize(self):
        dim = 32
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=1)
        vec = [math.sin(i) for i in range(dim)]
        qv = q.quantize(vec)
        assert qv.dimension == dim
        assert qv.ex_bits == 0
        assert qv.residual_norm > 0.0

    def test_4bit_quantize(self):
        dim = 32
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        vec = [math.sin(i) for i in range(dim)]
        qv = q.quantize(vec)
        assert qv.dimension == dim
        assert qv.ex_bits == 3
        assert len(qv.codes) == dim

    def test_approximate_distance_nonneg(self):
        dim = 32
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=1)
        vec = [1.0] * dim
        qv = q.quantize(vec)
        dist = q.approximate_distance([0.5] * dim, qv)
        assert dist >= 0.0

    def test_dimension_zero_rejected(self):
        with pytest.raises(ValueError):
            qntz.RaBitQQuantizer(0, 42)

    def test_bits_out_of_range(self):
        with pytest.raises(ValueError):
            qntz.RaBitQQuantizer(32, 42, total_bits=0)
        with pytest.raises(ValueError):
            qntz.RaBitQQuantizer(32, 42, total_bits=9)

    def test_fit_and_quantize(self):
        dim = 16
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        # 3 training vectors, flat
        data = [float(i) for i in range(dim * 3)]
        q.fit(data, 3)
        qv = q.quantize([1.0] * dim)
        assert qv.dimension == dim

    def test_quantize_dimension_mismatch(self):
        q = qntz.RaBitQQuantizer(8, 42)
        with pytest.raises(ValueError):
            q.quantize([1.0] * 4)

    def test_quantize_numpy(self):
        dim = 32
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=1)
        vec = np.array([math.sin(i) for i in range(dim)], dtype=np.float32)
        qv = q.quantize(vec)
        assert qv.dimension == dim

    def test_fit_2d_numpy(self):
        dim = 16
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        data = np.arange(dim * 3, dtype=np.float32).reshape(3, dim)
        q.fit(data)
        qv = q.quantize([1.0] * dim)
        assert qv.dimension == dim

    def test_fit_2d_numpy_matches_flat(self):
        """2D numpy fit produces the same result as flat list fit."""
        dim = 16
        flat_data = [float(i) for i in range(dim * 3)]
        np_data = np.array(flat_data, dtype=np.float32).reshape(3, dim)

        q1 = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        q1.fit(flat_data, 3)
        qv1 = q1.quantize([1.0] * dim)

        q2 = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        q2.fit(np_data)
        qv2 = q2.quantize([1.0] * dim)

        assert qv1.dimension == qv2.dimension
        assert abs(qv1.residual_norm - qv2.residual_norm) < 1e-6

    def test_fit_numpy_num_vectors_mismatch(self):
        dim = 16
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        data = np.arange(dim * 3, dtype=np.float32).reshape(3, dim)
        with pytest.raises(ValueError, match="num_vectors=5 does not match"):
            q.fit(data, num_vectors=5)

    def test_fit_flat_requires_num_vectors(self):
        dim = 16
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=4)
        with pytest.raises(ValueError, match="num_vectors is required"):
            q.fit([1.0] * (dim * 3))

    def test_approximate_distance_numpy_query(self):
        dim = 32
        q = qntz.RaBitQQuantizer(dim, 42, total_bits=1)
        qv = q.quantize([1.0] * dim)
        query = np.array([0.5] * dim, dtype=np.float32)
        dist = q.approximate_distance(query, qv)
        assert dist >= 0.0

    def test_repr(self):
        q = qntz.RaBitQQuantizer(32, 42, total_bits=4)
        assert repr(q) == "RaBitQQuantizer(dimension=32, total_bits=4)"

    def test_quantized_vector_repr(self):
        q = qntz.RaBitQQuantizer(32, 42, total_bits=1)
        qv = q.quantize([1.0] * 32)
        r = repr(qv)
        assert r.startswith("QuantizedVector(dimension=32")
        assert "ex_bits=0" in r


# ---------------------------------------------------------------------------
# Ternary
# ---------------------------------------------------------------------------


class TestTernary:
    def test_basic_quantize(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        assert tv.dimension == 4
        assert tv.get(0) == 1
        assert tv.get(1) == -1
        assert tv.get(2) == 0
        assert tv.get(3) == 0

    def test_to_list(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        assert tv.to_list() == [1, -1, 0, 0]

    def test_sparsity(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        assert 0.0 <= tv.sparsity <= 1.0
        # 2 out of 4 are zero -> 0.5
        assert abs(tv.sparsity - 0.5) < 1e-6

    def test_hamming(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        a = tq.quantize([0.5, -0.5, 0.0, 0.0])
        b = tq.quantize([0.5, 0.5, 0.0, -0.5])
        assert qntz.ternary_hamming(a, b) == 2

    def test_hamming_dimension_mismatch(self):
        tq4 = qntz.TernaryQuantizer(4, normalize=False)
        tq8 = qntz.TernaryQuantizer(8, normalize=False)
        a = tq4.quantize([1.0] * 4)
        b = tq8.quantize([1.0] * 8)
        assert qntz.ternary_hamming(a, b) is None

    def test_inner_product(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        a = tq.quantize([0.5, 0.5, 0.0, 0.0])
        b = tq.quantize([0.5, -0.5, 0.0, 0.0])
        ip = qntz.ternary_inner_product(a, b)
        # a = [1, 1, 0, 0], b = [1, -1, 0, 0] -> 1*1 + 1*(-1) = 0
        assert ip == 0

    def test_cosine_similarity(self):
        tq = qntz.TernaryQuantizer(8, normalize=False)
        a = tq.quantize([1.0] * 8)
        sim = qntz.ternary_cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_asymmetric_ip(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        # tv = [1, -1, 0, 0]
        query = [2.0, 3.0, 1.0, 1.0]
        ip = qntz.ternary_asymmetric_inner_product(query, tv)
        # 2*1 + 3*(-1) + 1*0 + 1*0 = -1
        assert abs(ip - (-1.0)) < 1e-6

    def test_asymmetric_cosine_distance(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, 0.5, 0.0, 0.0])
        # tv = [1, 1, 0, 0]
        query = [1.0, 1.0, 0.0, 0.0]
        dist = qntz.ternary_asymmetric_cosine_distance(query, tv)
        # cos = (1+1) / (sqrt(2) * sqrt(2)) = 1.0 -> distance = 0.0
        assert abs(dist) < 1e-6

    def test_quantize_dimension_mismatch(self):
        tq = qntz.TernaryQuantizer(8)
        with pytest.raises(ValueError):
            tq.quantize([1.0] * 4)

    def test_fit(self):
        dim = 8
        tq = qntz.TernaryQuantizer(dim, target_sparsity=0.5, normalize=False)
        data = [float(i) for i in range(dim * 10)]
        tq.fit(data, 10)
        tv = tq.quantize([0.5] * dim)
        assert tv.dimension == dim

    def test_quantize_numpy(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        vec = np.array([0.5, -0.5, 0.1, -0.1], dtype=np.float32)
        tv = tq.quantize(vec)
        assert tv.to_list() == [1, -1, 0, 0]

    def test_fit_2d_numpy(self):
        dim = 8
        tq = qntz.TernaryQuantizer(dim, target_sparsity=0.5, normalize=False)
        data = np.arange(dim * 10, dtype=np.float32).reshape(10, dim)
        tq.fit(data)
        tv = tq.quantize([0.5] * dim)
        assert tv.dimension == dim

    def test_fit_2d_numpy_matches_flat(self):
        dim = 8
        flat_data = [float(i) for i in range(dim * 10)]
        np_data = np.array(flat_data, dtype=np.float32).reshape(10, dim)

        tq1 = qntz.TernaryQuantizer(dim, target_sparsity=0.5, normalize=False)
        tq1.fit(flat_data, 10)
        tv1 = tq1.quantize([0.5] * dim)

        tq2 = qntz.TernaryQuantizer(dim, target_sparsity=0.5, normalize=False)
        tq2.fit(np_data)
        tv2 = tq2.quantize([0.5] * dim)

        assert tv1.to_list() == tv2.to_list()

    def test_asymmetric_ip_numpy(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        query = np.array([2.0, 3.0, 1.0, 1.0], dtype=np.float32)
        ip = qntz.ternary_asymmetric_inner_product(query, tv)
        assert abs(ip - (-1.0)) < 1e-6

    def test_asymmetric_cosine_distance_numpy(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, 0.5, 0.0, 0.0])
        query = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        dist = qntz.ternary_asymmetric_cosine_distance(query, tv)
        assert abs(dist) < 1e-6

    def test_repr(self):
        tq = qntz.TernaryQuantizer(8)
        assert repr(tq) == "TernaryQuantizer(dimension=8)"

    def test_vector_repr(self):
        tq = qntz.TernaryQuantizer(4, threshold_high=0.3, threshold_low=-0.3, normalize=False)
        tv = tq.quantize([0.5, -0.5, 0.1, -0.1])
        r = repr(tv)
        assert r.startswith("TernaryVector(dimension=4")
        assert "sparsity=" in r
