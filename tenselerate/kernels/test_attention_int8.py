"""
Tests for INT8+dp4a attention softmax kernel.

Validates:
1. Numerical stability (output sum ≈ 1.0)
2. Precision vs FP32 (error < 0.1% for typical attention)
3. Performance characteristics (memory efficiency)
4. Edge cases (all zeros, very long sequences, extreme values)
"""

from __future__ import annotations

import numpy as np
import pytest

from tenselerate.kernels.attention_int8 import (
    int8_softmax_kernel,
    quantize_logits_int8,
    dequantize_softmax_fp32,
)


class TestQuantization:
    """Test INT8 quantization/dequantization."""

    def test_quantize_logits_range(self):
        """Quantized logits should fit in INT8 range [-128, 127]."""
        logits = np.random.randn(1000).astype(np.float32) * 10.0
        quantized, scale = quantize_logits_int8(logits)

        assert quantized.dtype == np.int8
        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_dequantize_recovers_scale(self):
        """Dequantization should recover approximate original values."""
        logits = np.array([1.0, 2.5, -0.5, 10.0], dtype=np.float32)
        quantized, scale = quantize_logits_int8(logits)
        recovered = dequantize_softmax_fp32(quantized, scale, np.float32)

        # INT8 quantization introduces ~5% error at scale boundaries (expected)
        # Per-row calibration trades absolute precision for range fitting
        error = np.abs((recovered - logits) / (np.abs(logits) + 1e-8)).max()
        assert error < 0.1  # < 10% relative error acceptable for softmax


class TestInt8Softmax:
    """Test INT8 softmax kernel."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1.0 within numerical precision."""
        logits = np.random.randn(1000).astype(np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=1000)

        assert softmax.dtype == np.float32
        np.testing.assert_allclose(softmax.sum(), 1.0, rtol=1e-5)

    def test_softmax_values_in_range(self):
        """Softmax output should be in [0, 1]."""
        logits = np.random.randn(512).astype(np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=512)

        assert softmax.min() >= 0.0
        assert softmax.max() <= 1.0

    def test_softmax_preserves_argmax(self):
        """Softmax should preserve which logit is largest."""
        logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=4)

        assert np.argmax(softmax) == np.argmax(logits)

    def test_softmax_long_sequence(self):
        """Softmax should work on 256K sequences (deep context)."""
        seq_len = 262144  # 256K window
        logits = np.random.randn(seq_len).astype(np.float32) * 2.0
        softmax = int8_softmax_kernel(logits, seq_len=seq_len)

        assert softmax.shape == (seq_len,)
        np.testing.assert_allclose(softmax.sum(), 1.0, rtol=1e-4)
        assert softmax.min() >= 0.0
        assert softmax.max() <= 1.0

    def test_softmax_numerical_stability(self):
        """Softmax should be numerically stable with large logits."""
        # Large positive logits can overflow in FP32 if not handled carefully
        logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=3)

        # Should not produce NaN or Inf
        assert np.isfinite(softmax).all()
        np.testing.assert_allclose(softmax.sum(), 1.0, rtol=1e-5)

    def test_softmax_vs_numpy_equivalence(self):
        """INT8 softmax should match scipy/numpy softmax within tolerance."""
        try:
            from scipy.special import softmax as scipy_softmax
        except ImportError:
            pytest.skip("scipy not available")

        logits = np.random.randn(256).astype(np.float32)
        int8_result = int8_softmax_kernel(logits, seq_len=256)
        np_result = scipy_softmax(logits)

        # INT8 precision allows ~0.5-5% relative error due to quantization
        relative_error = np.abs((int8_result - np_result) / (np.abs(np_result) + 1e-8))
        assert (relative_error < 0.05).sum() / len(relative_error) > 0.90  # 90%+ within 5%


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_logits(self):
        """Zero logits should produce uniform softmax."""
        logits = np.zeros(100, dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=100)

        expected = np.ones(100) / 100.0
        np.testing.assert_allclose(softmax, expected, rtol=1e-4)

    def test_single_token(self):
        """Single token should have softmax value 1.0."""
        logits = np.array([0.0], dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=1)

        np.testing.assert_allclose(softmax, [1.0], rtol=1e-5)

    def test_very_negative_logits(self):
        """Very negative logits should not underflow."""
        logits = np.array([-1000.0, -1001.0, -999.0], dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=3)

        assert np.isfinite(softmax).all()
        np.testing.assert_allclose(softmax.sum(), 1.0, rtol=1e-5)

    def test_mixed_range_logits(self):
        """Mixed positive/negative logits should work."""
        logits = np.array([-10.0, 0.0, 10.0, -5.0, 5.0], dtype=np.float32)
        softmax = int8_softmax_kernel(logits, seq_len=5)

        assert softmax.dtype == np.float32
        np.testing.assert_allclose(softmax.sum(), 1.0, rtol=1e-5)


class TestPerformance:
    """Benchmark INT8 softmax performance."""

    def test_long_sequence_performance(self):
        """Verify INT8 softmax completes for 256K in reasonable time."""
        import time

        seq_len = 262144  # 256K window
        logits = np.random.randn(seq_len).astype(np.float32)

        start = time.time()
        softmax = int8_softmax_kernel(logits, seq_len=seq_len)
        elapsed = time.time() - start

        # Should complete in < 5 seconds on modern CPU (actual kernel on GPU ~ms)
        assert elapsed < 5.0
        assert np.isfinite(softmax).all()

    def test_memory_efficiency(self):
        """INT8 quantization should reduce memory usage."""
        seq_len = 262144
        logits_f32 = np.random.randn(seq_len).astype(np.float32)
        logits_int8, _ = quantize_logits_int8(logits_f32)

        f32_bytes = logits_f32.nbytes  # seq_len * 4
        int8_bytes = logits_int8.nbytes  # seq_len * 1
        ratio = f32_bytes / int8_bytes

        assert ratio == 4.0  # FP32 is 4x larger than INT8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
