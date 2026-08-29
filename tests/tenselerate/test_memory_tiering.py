"""
The sizing foundation for whole-system memory tiering (see
docs/tenselerate-memory-tiering.md). The GDN recurrent state is the engine's
real long-range memory and, unlike KV, does not scale with context or window;
a parked session's footprint is that state plus one window of KV, and it is a
constant - which is what lets a host-RAM tier hold a deterministic number of
live long-context sessions.
"""
from __future__ import annotations

import dataclasses

from tenselerate.config import RAVENX_27B

GiB = 1024 ** 3
MB = 1024 ** 2


def test_gdn_state_is_about_75_mb_at_bf16():
    mb = RAVENX_27B.gdn_state_bytes(2.0) / MB
    assert 70 < mb < 80, mb


def test_gdn_state_is_independent_of_context_and_window():
    """The whole point: the long-range memory does not grow with either."""
    base = RAVENX_27B.gdn_state_bytes()
    narrow = dataclasses.replace(RAVENX_27B, attention_window=32_768)
    wide = dataclasses.replace(RAVENX_27B, attention_window=131_072)
    assert narrow.gdn_state_bytes() == base == wide.gdn_state_bytes()


def test_gdn_state_precision_scales_linearly():
    assert RAVENX_27B.gdn_state_bytes(4.0) == 2 * RAVENX_27B.gdn_state_bytes(2.0)


def test_parked_footprint_is_window_kv_plus_state():
    foot = RAVENX_27B.parked_footprint_bytes()
    expect = (RAVENX_27B.kv_bytes_for_context(RAVENX_27B.resident_kv_tokens)
              + RAVENX_27B.gdn_state_bytes())
    assert foot == expect
    # dominated by the window KV (4.25 GiB at the default 128K window), with
    # the GDN state a real but small slice on top
    assert foot > RAVENX_27B.kv_bytes_for_context(RAVENX_27B.resident_kv_tokens)
    # at the 32K quality-floor window the footprint is ~1.14 GiB/session
    narrow = dataclasses.replace(RAVENX_27B, attention_window=32_768)
    assert 1.0 * GiB < narrow.parked_footprint_bytes() < 1.25 * GiB


def test_host_ram_parks_a_deterministic_session_count():
    narrow = dataclasses.replace(RAVENX_27B, attention_window=32_768)
    foot = narrow.parked_footprint_bytes()
    # a 256 GiB host parks a couple hundred live 1M-token sessions
    parked = int(256 * GiB / foot)
    assert 150 < parked < 300, parked


def test_state_only_cold_footprint_is_far_smaller_than_parked():
    """The L3 recompute-on-wake option: store state + tokens, not window KV."""
    state_only = RAVENX_27B.gdn_state_bytes()
    parked = RAVENX_27B.parked_footprint_bytes()
    assert state_only < parked / 10      # ~75 MB vs ~1.14 GiB
