"""
The 600 tok/s speed floor. Like the context floor and the model lock it is a
hard product limit: `tenselerate plan` refuses (exit 3) any machine that cannot
reach MIN_DECODE_TOKS of aggregate decode at the context floor at ANY attention
window. The window is the only dial - context never drops to buy speed.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from tenselerate.cli import main
from tenselerate.config import MIN_DECODE_TOKS


def run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_speed_floor_is_600():
    assert MIN_DECODE_TOKS == 600


def test_info_reports_the_speed_floor():
    rc, out = run(["info"])
    assert rc == 0
    assert f"SPEED FLOOR      : {MIN_DECODE_TOKS:,} tok/s" in out


def test_cmp_meets_the_floor_at_a_narrower_window():
    # at the default 128K window the CMP cannot reach 600, but a narrower
    # window can - so plan succeeds and shows the window that gets there
    rc, out = run(["plan", "--machine", "cmp170hx"])
    assert rc == 0
    assert f"reaches {MIN_DECODE_TOKS}+" in out
    assert "never context length" in out


def test_consumer_box_is_refused_not_served_slowly():
    # 24 GiB / ~500 GB/s cannot reach 600 tok/s at any window once the
    # weights are resident - the plan must refuse, not shrug
    rc, out = run(["plan", "--machine", "5070+3060"])
    assert rc == 3
    assert "NOT reachable" in out
    # and it must NOT have offered to lower the context to get there
    assert "never context length" in out


def test_low_bandwidth_cards_cannot_fake_the_floor():
    # the 3060 alone cannot even hold the weights; whatever exit path it
    # takes, it must never claim the speed floor is met
    rc, out = run(["plan", "--machine", "3060"])
    assert rc != 0
    assert "met at this window" not in out
