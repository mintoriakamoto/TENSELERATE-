"""
The quality floor. Speed is bought with a narrower attention window, and the
two previous floors (1M context, 600 tok/s) both push in that direction - this
floor is the stop: the window never drops below MIN_ATTENTION_WINDOW, and RoPE
scaling stays banned. 32K is the narrowest window that still meets the speed
floor on the target machine, so all three floors are satisfiable at once - by
design, and pinned here.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from tenselerate.cli import BW_EFFICIENCY, MACHINE_HW, main
from tenselerate.config import (
    DEFAULT_ATTENTION_WINDOW, MIN_ATTENTION_WINDOW, MIN_DECODE_TOKS,
    RAVENX_27B, QualityFloorError, validate_window,
)

GiB = 1024 ** 3


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_quality_floor_is_32k():
    assert MIN_ATTENTION_WINDOW == 32_768


def test_default_window_is_at_or_above_the_floor():
    assert DEFAULT_ATTENTION_WINDOW >= MIN_ATTENTION_WINDOW


def test_validate_window_enforces_the_floor():
    assert validate_window(MIN_ATTENTION_WINDOW) == MIN_ATTENTION_WINDOW
    assert validate_window(65_536) == 65_536
    for w in (16_384, 8_192, 1):
        with pytest.raises(QualityFloorError, match="quality"):
            validate_window(w)


def test_plan_refuses_a_window_below_the_floor():
    rc, out = run(["plan", "--machine", "cmp170hx",
                   "--attention-window", "16384"])
    assert rc == 2
    assert "quality" in out


def test_plan_accepts_the_floor_window_and_meets_the_speed_floor():
    rc, out = run(["plan", "--machine", "cmp170hx",
                   "--attention-window", str(MIN_ATTENTION_WINDOW)])
    assert rc == 0
    assert "met at this window" in out


def test_plan_never_offers_a_window_below_the_floor():
    rc, out = run(["plan", "--machine", "cmp170hx"])
    assert rc == 0
    assert "window  16,384" not in out and "window   8,192" not in out
    assert "quality floor" in out


def test_all_three_floors_are_simultaneously_satisfiable():
    """At the quality-floor window, the target machine reaches the speed
    floor at the context floor - the floors cannot deadlock each other."""
    vram, bw = MACHINE_HW["cmp170hx"]
    weights, overhead = 15.41, 1.5
    kv = RAVENX_27B.kv_bytes_per_token() * MIN_ATTENTION_WINDOW / GiB
    b = int((vram - weights - overhead) // kv)
    agg = bw * BW_EFFICIENCY / ((weights + kv * b) * 1.074) * b
    assert b >= 1
    assert agg >= MIN_DECODE_TOKS, agg


def test_info_reports_the_quality_floor():
    rc, out = run(["info"])
    assert rc == 0
    assert f"QUALITY FLOOR    : window >= {MIN_ATTENTION_WINDOW:,}" in out
    assert "no RoPE scaling" in out
