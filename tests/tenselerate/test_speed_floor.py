"""
The 400 tok/s speed floor. It is the product STANDARD, and it stays 400 even
though the one supported box (dual RTX 2080 Ti, 22 GiB) tops out ~152 tok/s
and so sits below it. The bar is not lowered to flatter the hardware: `plan`
reports the box as below standard (exit 3) rather than pretending. Serving
still works - only `plan` gates.
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


def test_speed_floor_is_400():
    assert MIN_DECODE_TOKS == 400


def test_info_reports_the_speed_floor():
    rc, out = run(["info"])
    assert rc == 0
    assert f"SPEED FLOOR      : {MIN_DECODE_TOKS:,} tok/s" in out


def test_supported_box_is_below_the_standard_and_says_so():
    # the dual 2080 Ti is the ONLY supported box and cannot reach 400 at any
    # window; plan reports that honestly (exit 3) instead of lowering the bar
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert rc == 3
    assert "BELOW" in out
    assert f"{MIN_DECODE_TOKS}" in out
    # and it must NOT drop context to chase speed
    assert "never context length" in out


def test_plan_defaults_to_the_only_box_at_the_context_floor():
    # default --machine is the one supported box; context floor still shown
    rc, out = run(["plan"])
    assert rc == 3                       # below the speed standard, honestly
    assert f"{MIN_CONTEXT_TOKENS:,} tokens" in out
    assert "2x2080ti" in out


def test_no_window_on_the_box_reaches_the_floor():
    # every offered window is below 400 - the ladder never flags "reaches 400+"
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert f"reaches {MIN_DECODE_TOKS}+" not in out
