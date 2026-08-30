"""
The acceleration dials in `plan`: the two real levers that speed decode on the
fixed dual 2080 Ti box - q4_0 KV cache (~2x concurrency) and MTP self-
speculation (~1.8x, identical output). Baseline is below the 400 standard; the
two together model past it. Roofline, honestly labelled - q4_0 is a quality
trade, MTP is a roadmap kernel.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from tenselerate.cli import main
from tenselerate.config import (
    KV_BITS_PER_ELEM, MIN_DECODE_TOKS, MTP_SPECULATIVE_SPEEDUP,
)


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_constants_are_sane():
    assert KV_BITS_PER_ELEM[8] > KV_BITS_PER_ELEM[4]         # q4_0 is smaller
    assert abs(KV_BITS_PER_ELEM[4] / KV_BITS_PER_ELEM[8] - 0.53) < 0.05
    assert 1.5 <= MTP_SPECULATIVE_SPEEDUP <= 2.5


def test_baseline_is_below_but_shows_the_acceleration_path():
    rc, out = run(["plan", "--machine", "2x2080ti"])
    assert rc == 3
    assert "BELOW the standard" in out
    assert "acceleration path" in out
    assert "q4_0 KV" in out and "MTP spec" in out
    assert f"REACHES the {MIN_DECODE_TOKS} standard" in out


def test_full_stack_reaches_the_standard():
    # q4_0 KV + MTP together clear the 400 floor at the 32K quality-floor window
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--kv-bits", "4", "--spec", "mtp"])
    assert rc == 0
    assert "acceleration     : KV q4_0 + MTP spec" in out
    assert f"reaches {MIN_DECODE_TOKS}+" in out


def test_either_lever_alone_is_not_enough():
    # neither q4_0 KV nor MTP on its own reaches 400 on this box - it takes both
    rc4, _ = run(["plan", "--machine", "2x2080ti", "--kv-bits", "4"])
    rcm, _ = run(["plan", "--machine", "2x2080ti", "--spec", "mtp"])
    assert rc4 == 3 and rcm == 3


def test_mtp_multiplies_the_baseline_throughput():
    # MTP is a pure multiplier: the batch-1 decode rate scales by ~1.8x
    _, base = run(["plan", "--machine", "2x2080ti"])
    _, spec = run(["plan", "--machine", "2x2080ti", "--spec", "mtp"])

    def batch1(out: str) -> float:
        line = next(ln for ln in out.splitlines()
                    if ln.startswith("decode (batch 1)"))
        return float(line.split("~")[1].split()[0].replace(",", ""))
    assert batch1(spec) > batch1(base) * 1.5


def test_context_floor_still_holds_under_acceleration():
    # acceleration never trades away context - it is still the 1M floor
    rc, out = run(["plan", "--machine", "2x2080ti",
                   "--kv-bits", "4", "--spec", "mtp", "--ctx", "8192"])
    assert rc == 2                       # below the context floor, refused
    assert "below the TENSELERATE floor" in out
