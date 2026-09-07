"""
The quality floor is now a LOCK: the attention window is pinned at the maximum
no-RoPE recall (the trained rotary range minus the sinks, 262,140), and it never
narrows for speed. MIN == MAX == the ceiling, so the only legal window is that
one value. Verbatim recall is pinned at its deepest; the box takes whatever
throughput that leaves, below the speed target by design. No RoPE scaling, ever.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from tenselerate.cli import main
from tenselerate.config import (
    ATTENTION_SINK_TOKENS, DEFAULT_ATTENTION_WINDOW, MAX_ATTENTION_WINDOW,
    MIN_ATTENTION_WINDOW, QWEN38_27B, QualityFloorError,
    RopeScalingRequired, validate_window,
)


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_the_window_is_locked_at_the_max_recall():
    # min == max == the ceiling: one legal window, the deepest no-RoPE recall
    assert MIN_ATTENTION_WINDOW == MAX_ATTENTION_WINDOW == 262_140
    assert DEFAULT_ATTENTION_WINDOW == MIN_ATTENTION_WINDOW
    assert MAX_ATTENTION_WINDOW == QWEN38_27B.max_position_embeddings - ATTENTION_SINK_TOKENS


def test_validate_window_only_accepts_the_locked_value():
    assert validate_window(MIN_ATTENTION_WINDOW) == MIN_ATTENTION_WINDOW
    # anything narrower is refused - the window does not narrow for speed
    for w in (131_072, 65_536, 32_768, 1):
        with pytest.raises(QualityFloorError, match="LOCKED"):
            validate_window(w)
    # anything wider needs RoPE scaling, which we never do
    for w in (MAX_ATTENTION_WINDOW + 1, 500_000, 1_000_000):
        with pytest.raises(RopeScalingRequired, match="never scales RoPE"):
            validate_window(w)


def test_plan_runs_at_the_locked_window_by_default():
    # no --attention-window: the default IS the locked 262,140; it fits the
    # Ampere box and is honestly below the speed target (exit 3, not refused)
    rc, out = run(["plan", "--machine", "cmp170hx+3060", "--kv-bits", "4"])
    assert rc == 3
    assert "FITS" in out and "DOES NOT FIT" not in out
    assert f"locked at {MAX_ATTENTION_WINDOW:,}" in out


def test_plan_refuses_a_narrower_window():
    # the window cannot be narrowed for speed anymore - exit 2 (quality lock)
    rc, out = run(["plan", "--machine", "cmp170hx+3060",
                   "--attention-window", "131072"])
    assert rc == 2
    assert "LOCKED" in out


def test_plan_refuses_a_window_above_the_ceiling():
    rc, out = run(["plan", "--machine", "cmp170hx+3060",
                   "--attention-window", str(MAX_ATTENTION_WINDOW + 1)])
    assert rc == 2
    assert "never scales RoPE" in out


def test_plan_never_offers_a_narrower_window():
    # the old window ladder is gone - no narrower windows are ever suggested
    _, out = run(["plan", "--machine", "cmp170hx+3060"])
    assert "window  32,768" not in out and "window  65,536" not in out
    assert "narrows for speed" in out


def test_info_reports_the_locked_window():
    rc, out = run(["info"])
    assert rc == 0
    assert f"QUALITY FLOOR    : window LOCKED at {MIN_ATTENTION_WINDOW:,}" in out
    assert "no RoPE scaling" in out
