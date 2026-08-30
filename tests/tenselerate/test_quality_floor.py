"""
The 32K quality floor: the narrowest attention window the engine will run, and
no RoPE scaling. It is machine-independent. On the one supported box (dual
2080 Ti) the speed floor is NOT met even at this window, so the context, speed,
and quality floors are NOT all simultaneously satisfiable there - the box is
below the speed standard by design, and the engine says so rather than
loosening quality to compensate.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from tenselerate.cli import BW_EFFICIENCY, MACHINE_HW, main
from tenselerate.config import (
    ATTENTION_SINK_TOKENS, DEFAULT_ATTENTION_WINDOW, MAX_ATTENTION_WINDOW,
    MIN_ATTENTION_WINDOW, MIN_DECODE_TOKS, RAVENX_27B, QualityFloorError,
    RopeScalingRequired, validate_window,
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


def test_window_ceiling_is_the_rotary_range_minus_the_sinks():
    # the deepest no-RoPE window: trained rotary range minus the sinks that
    # share it, so window + sinks never extrapolates past the range
    assert MAX_ATTENTION_WINDOW == RAVENX_27B.max_position_embeddings - ATTENTION_SINK_TOKENS
    assert MAX_ATTENTION_WINDOW == 262_140
    assert RAVENX_27B.max_attention_window == MAX_ATTENTION_WINDOW


def test_validate_window_enforces_the_ceiling():
    # at the ceiling is fine; one token past it needs RoPE scaling -> refused
    assert validate_window(MAX_ATTENTION_WINDOW) == MAX_ATTENTION_WINDOW
    for w in (MAX_ATTENTION_WINDOW + 1, 300_000, 1_000_000):
        with pytest.raises(RopeScalingRequired, match="never scales RoPE"):
            validate_window(w)


def test_plan_runs_at_the_262k_ceiling_with_q4_kv():
    # the user's deepest-recall config: 262,140 window fits the 22 GiB box only
    # with q4_0 KV, and is honestly below the speed floor (exit 3, not refused)
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--attention-window", str(MAX_ATTENTION_WINDOW),
                   "--kv-bits", "4"])
    assert rc == 3
    assert "FITS" in out
    assert "DOES NOT FIT" not in out


def test_plan_refuses_a_window_above_the_ceiling():
    # past the ceiling is a no-RoPE violation, checked before the roofline
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--attention-window", str(MAX_ATTENTION_WINDOW + 1)])
    assert rc == 2
    assert "never scales RoPE" in out


def test_262k_ceiling_needs_q4_to_fit_the_box():
    # at q8_0 the 262K window's KV does not fit 22 GiB - the ceiling is a
    # q4_0-only config on this box, exactly as info says
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--attention-window", str(MAX_ATTENTION_WINDOW)])
    assert "DOES NOT FIT" in out
    assert rc == 1


def test_info_reports_the_window_ceiling():
    rc, out = run(["info"])
    assert rc == 0
    assert f"WINDOW CEILING   : window <= {MAX_ATTENTION_WINDOW:,}" in out


def test_plan_refuses_a_window_below_the_floor():
    # quality is checked before the speed roofline, so a sub-floor window is
    # exit 2 (quality) regardless of the machine
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--attention-window", "16384"])
    assert rc == 2
    assert "quality" in out


def test_floor_window_passes_quality_but_box_still_misses_speed():
    # at the 32K quality-floor window the window is accepted (no quality error),
    # but the supported box still cannot reach 400 -> exit 3, not exit 2
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--attention-window", str(MIN_ATTENTION_WINDOW)])
    assert rc == 3                       # below speed, not a quality refusal
    assert "below the TENSELERATE quality floor" not in out


def test_plan_never_offers_a_window_below_the_floor():
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert "window  16,384" not in out and "window   8,192" not in out
    assert "quality floor" in out


def test_floors_are_NOT_all_satisfiable_on_the_supported_box():
    """The honest inversion: on the only supported box the speed floor is not
    met even at the quality-floor window, so context+speed+quality do not all
    hold at once. The bar stays; the box is reported below it."""
    vram, bw = MACHINE_HW["2x2080ti"]
    weights, overhead = 15.41, 1.5
    kv = RAVENX_27B.kv_bytes_per_token() * MIN_ATTENTION_WINDOW / GiB
    b = int((vram - weights - overhead) // kv)
    agg = bw * BW_EFFICIENCY / ((weights + kv * b) * 1.074) * b
    assert b >= 1                        # it does serve (context floor holds)
    assert agg < MIN_DECODE_TOKS         # ...but below the speed standard


def test_info_reports_the_quality_floor():
    rc, out = run(["info"])
    assert rc == 0
    assert f"QUALITY FLOOR    : window >= {MIN_ATTENTION_WINDOW:,}" in out
    assert "no RoPE scaling" in out
