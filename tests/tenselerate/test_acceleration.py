"""
With the window locked at max recall, speed is no longer bought by narrowing it.
The only levers left are lossless - q4_0 KV (more concurrency) and MTP self-
speculation (~1.8x, identical output; EAGLE-3 higher). `plan` models them at the
LOCKED window and reports honestly that they do NOT reach the 400 target there,
because quality is pinned at maximum. Roofline, honestly labelled.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from tenselerate.cli import main
from tenselerate.config import (
    KV_BITS_PER_ELEM, MIN_DECODE_TOKS, MTP_SPECULATIVE_SPEEDUP,
)

BOX = ["--machine", "cmp170hx+3060"]


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_constants_are_sane():
    assert KV_BITS_PER_ELEM[8] > KV_BITS_PER_ELEM[4]         # q4_0 is smaller
    assert abs(KV_BITS_PER_ELEM[4] / KV_BITS_PER_ELEM[8] - 0.53) < 0.05
    assert 1.5 <= MTP_SPECULATIVE_SPEEDUP <= 2.5


def test_box_is_below_target_and_shows_the_lossless_levers():
    rc, out = run(["plan", *BOX, "--kv-bits", "4", "--spec", "mtp"])
    assert rc == 3
    assert "BELOW it" in out
    assert "lossless levers at the locked window" in out
    assert "quality-over-speed" in out


def test_even_the_full_stack_stays_under_the_target():
    # q4_0 KV + MTP at the locked window do NOT reach 400 - quality is pinned
    _, out = run(["plan", *BOX, "--kv-bits", "4", "--spec", "mtp"])
    assert f"still under the {MIN_DECODE_TOKS} target" in out


def test_mtp_multiplies_the_baseline_throughput():
    # MTP is a pure multiplier: the batch-1 decode rate scales by ~1.8x
    _, base = run(["plan", *BOX])
    _, spec = run(["plan", *BOX, "--spec", "mtp"])

    def batch1(out: str) -> float:
        line = next(ln for ln in out.splitlines()
                    if ln.startswith("decode (batch 1)"))
        return float(line.split("~")[1].split()[0].replace(",", ""))
    assert batch1(spec) > batch1(base) * 1.5


def test_context_floor_still_holds_under_acceleration():
    # acceleration never trades away context - it is still the 1M floor
    rc, out = run(["plan", *BOX, "--kv-bits", "4", "--spec", "mtp",
                   "--ctx", "8192"])
    assert rc == 2                       # below the context floor, refused
    assert "below the TENSELERATE floor" in out
