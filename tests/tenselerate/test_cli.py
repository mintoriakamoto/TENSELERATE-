"""
CLI behaviour: the subcommands exist, the 1M floor is enforced at the command
level, and `plan` never reports a batch size that could not fit in VRAM.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from tenselerate.cli import build_parser, main
from tenselerate.config import MIN_CONTEXT_TOKENS


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_all_subcommands_parse():
    """Each documented subcommand is accepted by the parser (no private attrs)."""
    ap = build_parser()
    for cmd in ("update", "info", "plan", "doctor", "serve"):
        ns = ap.parse_args([cmd])
        assert ns.command == cmd
        assert callable(ns.func)


def test_unlisted_subcommand_is_rejected():
    ap = build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["definitely-not-a-command"])


def test_info_reports_the_floor_and_no_rope_scaling():
    rc, out = run(["info"])
    assert rc == 0
    assert f"{MIN_CONTEXT_TOKENS:,}" in out
    assert "rope scaling: no" in out
    # every listed context must be at or above the floor
    assert "4,000,000" in out and "10,000,000" in out


def test_plan_defaults_to_the_floor():
    rc, out = run(["plan", "--machine", "cmp170hx"])
    assert rc == 0
    assert f"{MIN_CONTEXT_TOKENS:,} tokens" in out


def test_plan_rejects_context_below_the_floor():
    rc, out = run(["plan", "--machine", "cmp170hx", "--ctx", "8192"])
    assert rc == 2
    assert "below the TENSELERATE floor" in out


def test_plan_never_reports_an_infeasible_batch():
    """
    Each concurrent sequence holds its own windowed KV, so batch is memory-capped.
    Parse the reported batches and check every one actually fits.
    """
    rc, out = run(["plan", "--machine", "cmp170hx"])
    assert rc == 0
    kv_line = next(ln for ln in out.splitlines() if "KV (windowed)" in ln)
    kv_gib = float(kv_line.split(":")[1].strip().split()[0])
    weights_line = next(ln for ln in out.splitlines() if ln.startswith("weights"))
    weights = float(weights_line.split(":")[1].strip().split()[0])
    vram, overhead = 64.0, 1.5

    batches = [int(ln.split("batch")[1].split(":")[0].strip())
               for ln in out.splitlines()
               if ln.strip().startswith("batch ")]
    assert batches, out
    for b in batches:
        assert weights + kv_gib * b + overhead <= vram, (b, out)


def test_plan_shows_a_route_to_600_when_the_default_window_cannot():
    rc, out = run(["plan", "--machine", "cmp170hx"])
    assert rc == 0
    if "600+" not in out.split("ceiling here")[0]:
        # it must then say what window would get there, at the same context
        assert "reaches 600+" in out
        assert "never context length" in out


def test_plan_on_the_consumer_box_still_holds_the_floor():
    rc, out = run(["plan", "--machine", "5070+3060"])
    assert rc in (0, 1)           # may not fit, but must not lower the context
    assert f"{MIN_CONTEXT_TOKENS:,} tokens" in out
    if rc == 1:
        assert "the floor is fixed" in out


def test_serve_refuses_off_host():
    rc, out = run(["serve", "--host", "0.0.0.0"])
    assert rc == 2
    assert "loopback-only" in out


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
