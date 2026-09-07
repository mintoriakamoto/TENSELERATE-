"""
The 400 tok/s is now a TARGET, not a hard gate. The window is locked at max
recall, so the box (CMP 170HX + RTX 3060) runs below 400 by design - `plan`
reports the gap honestly (exit 3) and never narrows the window to chase it.
Quality won; speed takes what recall leaves.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from tenselerate.cli import main
from tenselerate.config import MIN_CONTEXT_TOKENS, MIN_DECODE_TOKS


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_speed_target_is_400():
    assert MIN_DECODE_TOKS == 400


def test_info_reports_the_speed_target():
    rc, out = run(["info"])
    assert rc == 0
    assert f"SPEED TARGET     : {MIN_DECODE_TOKS:,} tok/s" in out
    assert "NOT a hard gate" in out


def test_box_is_below_the_target_and_says_so():
    rc, out = run(["plan", "--machine", "cmp170hx+3060", "--kv-bits", "4",
                   "--spec", "mtp"])
    assert rc == 3
    assert "BELOW it" in out
    assert f"{MIN_DECODE_TOKS}" in out
    # it does not narrow the window to chase speed - quality is pinned
    assert "narrows for speed" in out


def test_plan_defaults_to_the_target_box_at_the_context_floor():
    rc, out = run(["plan", "--kv-bits", "4"])
    assert rc == 3                       # below the target, honestly
    assert f"{MIN_CONTEXT_TOKENS:,} tokens" in out
    assert "cmp170hx+3060" in out


def test_the_locked_window_never_reaches_the_target():
    # there is no window ladder anymore; nothing flags "reaches 400+"
    _, out = run(["plan", "--machine", "cmp170hx+3060", "--kv-bits", "4",
                  "--spec", "mtp"])
    assert f"reaches {MIN_DECODE_TOKS}+" not in out
