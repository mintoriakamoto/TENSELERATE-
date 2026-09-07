"""
The llama.cpp backend launcher: it builds the llama-server argv Hercules is
served with, shaped by the box's measurements, and refuses out-of-contract
requests before anything is launched. No GPU, no build needed.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from tenselerate.backends.llamacpp import (
    DEFAULT_CTX_POOL, DEFAULT_SLOTS, KV_TYPES, build_llama_server_argv,
    llama_server_command, llama_server_env,
)
from tenselerate.cli import main
from tenselerate.config import MIN_ATTENTION_WINDOW

MODEL = "/models/qwen3.8-27b-UD-Q4_K_M.gguf"


def argv(**kw) -> list[str]:
    return build_llama_server_argv(MODEL, **kw)


def after(a: list[str], flag: str) -> str:
    return a[a.index(flag) + 1]


def run(cli: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(cli)
    return rc, buf.getvalue()


def test_measured_defaults_land_in_the_argv():
    a = argv()
    assert a[0] == "llama-server" and after(a, "-m") == MODEL
    assert after(a, "-np") == str(DEFAULT_SLOTS) == "4"
    assert after(a, "-c") == str(DEFAULT_CTX_POOL)
    assert "--kv-unified" in a and "-cb" in a
    assert after(a, "-ctk") == "q8_0" and after(a, "-ctv") == "q8_0"
    assert after(a, "-ngl") == "999" and after(a, "-fa") == "on"
    assert after(a, "--cache-reuse") == "256"
    kwargs = json.loads(after(a, "--chat-template-kwargs"))
    assert kwargs == {"reasoning_effort": "low"}


def test_never_enables_speculation():
    # MTP measured 7-11% acceptance on the served merge: slower than plain
    a = argv()
    assert "--spec-type" not in a and "-md" not in a
    assert not any(x.startswith("--spec-draft") for x in a)


def test_pool_must_hold_one_locked_window():
    with pytest.raises(ValueError, match="locked"):
        argv(ctx_pool=MIN_ATTENTION_WINDOW - 1)
    assert after(argv(ctx_pool=MIN_ATTENTION_WINDOW), "-c") == str(MIN_ATTENTION_WINDOW)


def test_refuses_off_host_bad_kv_bad_reasoning_and_zero_slots():
    with pytest.raises(ValueError, match="loopback"):
        argv(host="0.0.0.0")
    with pytest.raises(ValueError, match="kv must be"):
        argv(kv="q6_K")
    with pytest.raises(ValueError, match="reasoning must be"):
        argv(reasoning="max")
    with pytest.raises(ValueError, match="slots"):
        argv(slots=0)
    assert set(KV_TYPES) == {"q8_0", "q4_0", "f16"}


def test_no_mmvq_is_an_environment_flag_not_an_argv_flag():
    base = {"PATH": "/usr/bin"}
    assert "GGML_CUDA_NO_MMVQ" not in llama_server_env(base=base)
    assert llama_server_env(no_mmvq=True, base=base)["GGML_CUDA_NO_MMVQ"] == "1"
    assert "GGML_CUDA_NO_MMVQ" not in " ".join(argv())
    assert llama_server_command(MODEL, no_mmvq=True).startswith("GGML_CUDA_NO_MMVQ=1 ")
    assert not llama_server_command(MODEL).startswith("GGML_CUDA")


def test_cli_serve_llamacpp_dry_run_prints_the_launch():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL, "--dry-run"])
    assert rc == 0
    assert "llama-server" in out and "--kv-unified" in out and "-np 4" in out
    assert "--spec-type" not in out
    assert "dry run" in out


def test_cli_serve_llamacpp_no_mmvq_shows_the_env_prefix():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--no-mmvq", "--slots", "2", "--kv", "q4_0", "--dry-run"])
    assert rc == 0
    assert "GGML_CUDA_NO_MMVQ=1 " in out and "-np 2" in out and "-ctk q4_0" in out


def test_cli_serve_llamacpp_needs_a_model_and_a_loopback_host():
    rc, out = run(["serve", "--backend", "llamacpp", "--dry-run"])
    assert rc == 2 and "--model" in out
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--host", "0.0.0.0", "--dry-run"])
    assert rc == 2


def test_cli_boot_can_bring_up_the_llamacpp_backend():
    # doctor runs first (no GPU here, so --force), then the same launch path
    rc, out = run(["boot", "--backend", "llamacpp", "--model", MODEL,
                   "--force", "--dry-run"])
    assert rc == 0
    assert "--kv-unified" in out
