"""
NVTX instrumentation.

The engine spec requires every new module to be instrumented with NVTX markers so
kernels are visible under Nsight Systems. This wraps NVTX when the CUDA bindings
are present and degrades to a zero-overhead no-op otherwise, so the same source
runs on a dev box with no CUDA and profiles on the real GPUs unchanged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

_ENABLED = os.environ.get("TENSELERATE_NVTX", "1") != "0"

try:  # pragma: no cover - exercised only where cuda-python is installed
    if _ENABLED:
        from nvtx import annotate as _annotate  # type: ignore
    else:  # pragma: no cover
        _annotate = None
except Exception:  # pragma: no cover - the common dev-box path
    _annotate = None


@contextmanager
def range(name: str, category: str = "tenselerate") -> Iterator[None]:
    """Mark a code range for Nsight. No-op when NVTX is unavailable."""
    if _annotate is None:
        yield
        return
    with _annotate(message=name, domain=category):  # pragma: no cover
        yield


def available() -> bool:
    return _annotate is not None
