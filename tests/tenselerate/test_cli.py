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
    for cmd in ("install", "build", "boot", "update", "info", "plan",
                "doctor", "serve"):
        ns = ap.parse_args([cmd])
        assert ns.command == cmd
        assert callable(ns.func)


def test_boot_refuses_off_host_before_building_or_serving():
    # boot binds a server, so it enforces the same loopback-only rule as serve
    rc, out = run(["boot", "--host", "0.0.0.0"])
    assert rc == 2
    assert "loopback-only" in out


def test_build_flags_are_mutually_exclusive():
    ap = build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["build", "--cpu", "--cuda"])


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


def test_plan_defaults_to_the_only_box_and_holds_the_context_floor():
    # default --machine is the one supported box (dual 2080 Ti); it is below
    # the speed standard (exit 3) but still serves the 1M context floor
    rc, out = run(["plan"])
    assert rc == 3
    assert "2x2080ti" in out
    assert f"{MIN_CONTEXT_TOKENS:,} tokens" in out


def test_plan_rejects_context_below_the_floor():
    rc, out = run(["plan", "--machine", "2x2080ti", "--ctx", "8192"])
    assert rc == 2
    assert "below the TENSELERATE floor" in out


def test_plan_never_reports_an_infeasible_batch():
    """
    Each concurrent sequence holds its own windowed KV, so batch is memory-capped.
    Parse the reported batches and check every one actually fits the 22 GiB box.
    """
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert rc == 3                       # below the speed standard, but valid
    kv_line = next(ln for ln in out.splitlines() if "KV (windowed)" in ln)
    kv_gib = float(kv_line.split(":")[1].strip().split()[0])
    weights_line = next(ln for ln in out.splitlines() if ln.startswith("weights"))
    weights = float(weights_line.split(":")[1].strip().split()[0])
    vram, overhead = 22.0, 1.5

    batches = [int(ln.split("batch")[1].split(":")[0].strip())
               for ln in out.splitlines()
               if ln.strip().startswith("batch ")]
    assert batches, out
    for b in batches:
        assert weights + kv_gib * b + overhead <= vram, (b, out)


def test_plan_reports_the_box_below_standard_without_dropping_context():
    from tenselerate.config import MIN_DECODE_TOKS
    # no window on the box reaches the floor; plan says so honestly and never
    # trades away context to chase speed
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert rc == 3
    assert f"reaches {MIN_DECODE_TOKS}+" not in out
    assert "BELOW" in out
    assert "never context length" in out


def test_serve_refuses_off_host():
    rc, out = run(["serve", "--host", "0.0.0.0"])
    assert rc == 2
    assert "loopback-only" in out


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
