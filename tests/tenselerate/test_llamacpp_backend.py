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


def test_agent_contract_flags_are_present():
    # Hermes: without --jinja llama-server ignores `tools`; reasoning_effort is
    # a template kwarg (needs jinja too); thinking must come back as
    # reasoning_content; an oversized request must fail, not context-shift.
    a = argv()
    assert "--jinja" in a
    assert after(a, "--reasoning-format") == "deepseek"
    assert "--no-context-shift" in a
    assert after(a, "--alias") == "tenselerate"
    assert after(argv(alias="hercules-27b"), "--alias") == "hercules-27b"
    with pytest.raises(ValueError, match="alias"):
        argv(alias="")


def test_speculation_is_off_by_default_and_depth_one_when_asked():
    # The merge left the MTP head's position-1 prediction intact and broke
    # positions 2+: n-max 1 measured +35%, n-max 5 measured -22%.
    a = argv()
    assert "--spec-type" not in a and "-md" not in a
    assert not any(x.startswith("--spec-draft") for x in a)
    b = argv(mtp_draft=1)
    assert after(b, "--spec-type") == "draft-mtp" and after(b, "--spec-draft-n-max") == "1"
    assert "-md" not in b            # the head is inside the -MTP- GGUF
    with pytest.raises(ValueError, match="mtp_draft"):
        argv(mtp_draft=-1)
    with pytest.raises(ValueError, match="mtp_draft"):
        argv(mtp_draft=99)


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


def test_cli_mtp_draft_flag():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mtp-draft", "1", "--dry-run"])
    assert rc == 0 and "--spec-type draft-mtp --spec-draft-n-max 1" in out


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
