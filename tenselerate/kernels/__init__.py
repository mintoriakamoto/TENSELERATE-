"""
Optimized kernels for TENSELERATE inference on heterogeneous Ampere GPUs.

This module houses compute kernels that target specific hardware strengths:
- INT8+dp4a on CMP 170HX (GA100): mining-optimized int8 matrix ops
- Mixed-precision softmax on RTX 3060 (GA106): tensor core efficiency
- Adaptive precision kernels for long-context attention

All kernels are designed for the 256K windowed attention architecture,
with special attention to bandwidth-constrained deep-context decoding.
"""

from tenselerate.kernels.attention_int8 import (
    int8_softmax_kernel,
    quantize_logits_int8,
    dequantize_softmax_fp32,
)

__all__ = [
    "int8_softmax_kernel",
    "quantize_logits_int8",
    "dequantize_softmax_fp32",
]
