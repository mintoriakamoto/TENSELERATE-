"""
The llama.cpp backend launcher: it builds the llama-server argv Hercules is
served with, shaped by the box's measurements, and refuses out-of-contract
requests before anything is launched. No GPU, no build needed.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from tenselerate.backends.llamacpp import (
    DEFAULT_CTX_POOL, DEFAULT_SLOTS, KV_TYPES, build_llama_server_argv,
    env_prefix, llama_server_command, llama_server_env, resolve_mtp_draft,
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
    assert after(a, "--reasoning-effort") == "low"       # first-class since the upstream sync
    assert "--chat-template-kwargs" not in a


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


def test_speculation_follows_the_gguf_and_depth_one_when_asked():
    # The merge left the MTP head's position-1 prediction intact and broke
    # positions 2+: n-max 1 measured +38% JSON / +35% code / +13% prose,
    # n-max 5 measured -22%. Default: depth 1 on an -MTP- GGUF, off otherwise.
    a = argv()                                   # MODEL has no MTP head
    assert "--spec-type" not in a and "-md" not in a
    assert not any(x.startswith("--spec-draft") for x in a)
    m = build_llama_server_argv("/models/Qwen3.8-27B-TurboFCFusion-MTP-Q4_K_M.gguf")
    assert after(m, "--spec-type") == "draft-mtp" and after(m, "--spec-draft-n-max") == "1"
    assert resolve_mtp_draft("x-MTP-Q4.gguf", None) == 1
    assert resolve_mtp_draft("x-Q4.gguf", None) == 0
    assert resolve_mtp_draft("x-MTP-Q4.gguf", 0) == 0    # explicit off wins
    b = argv(mtp_draft=1)
    assert after(b, "--spec-type") == "draft-mtp" and after(b, "--spec-draft-n-max") == "1"
    assert "-md" not in b            # the head is inside the -MTP- GGUF
    with pytest.raises(ValueError, match="mtp_draft"):
        argv(mtp_draft=-1)
    with pytest.raises(ValueError, match="mtp_draft"):
        argv(mtp_draft=99)


def test_greedy_sampling_is_the_server_default_because_acceptance_is_exact_match():
    # Production server, same head, same flag: greedy 46.2 tok/s at 88% draft
    # acceptance; the model card's temp 0.7 + repeat-penalty 1.15 gives 29.9 at
    # 22% - below the 34.4 no-MTP baseline. llama.cpp accepts a draft token only
    # when the sampled target token equals it, so sampling is a server default.
    a = argv()
    assert after(a, "--temp") == "0" and after(a, "--repeat-penalty") == "1.0"
    c = argv(sampling="client")
    assert "--temp" not in c and "--repeat-penalty" not in c
    # pure greedy loops in <think> on the merge; the two guards keep repeat
    # penalty off (it rewrites the argmax on every recent token)
    d = argv(sampling="dry")
    assert after(d, "--temp") == "0" and after(d, "--dry-multiplier") == "0.8"
    assert after(d, "--dry-penalty-last-n") == "2048"     # never scan the 262K window per token
    lo = argv(sampling="low")
    assert after(lo, "--temp") == "0.3" and after(lo, "--min-p") == "0.1"
    assert after(lo, "--repeat-penalty") == "1.0" and "--dry-multiplier" not in lo
    with pytest.raises(ValueError, match="sampling"):
        argv(sampling="warm")


def test_retrained_head_is_a_sidecar_at_depth_three():
    # scripts/mtp-head-train.py exports a re-aligned head as its own GGUF; it is
    # served with -md and, unless told otherwise, at the healthy-curve optimum n-max 3.
    a = argv(mtp_model="/models/mtp-head-q8_0.gguf")
    assert after(a, "-md") == "/models/mtp-head-q8_0.gguf"
    assert after(a, "--spec-type") == "draft-mtp" and after(a, "--spec-draft-n-max") == "3"
    b = argv(mtp_model="/m/head.gguf", mtp_draft=5)
    assert after(b, "--spec-draft-n-max") == "5"
    with pytest.raises(ValueError, match="mtp_model"):
        argv(mtp_model="/m/head.gguf", mtp_draft=0)
    assert "-md" not in argv(mtp_draft=1)
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mtp-model", "/m/head.gguf", "--dry-run"])
    assert rc == 0 and "-md /m/head.gguf" in out and "--spec-draft-n-max 3" in out


def test_prompt_cache_keeps_the_system_prefix_across_slots_and_restarts():
    # The ~35K Hermes system prompt is shared by the main loop and every
    # delegation child. RAM prompt cache + idle-slot caching + LCP slot
    # selection let a child inherit it instead of re-prefilling (~40 s);
    # --slot-save-path exposes save/restore so it survives a restart.
    a = argv()
    assert after(a, "--cache-ram") == "16384"
    assert "--cache-idle-slots" in a and "--no-cache-idle-slots" not in a
    assert after(a, "--slot-prompt-similarity") == "0.1"
    assert "--slot-save-path" not in a
    b = argv(cache_ram_mib=32768, cache_idle_slots=False, slot_similarity=0.5,
             slot_save_path="/var/lib/tenselerate/slots")
    assert after(b, "--cache-ram") == "32768" and "--no-cache-idle-slots" in b
    assert after(b, "--slot-prompt-similarity") == "0.5"
    assert after(b, "--slot-save-path") == "/var/lib/tenselerate/slots"
    with pytest.raises(ValueError, match="cache_idle_slots"):
        argv(cache_ram_mib=0)                       # idle-slot caching needs a cache
    assert "--cache-ram" in argv(cache_ram_mib=0, cache_idle_slots=False)
    with pytest.raises(ValueError, match="slot_similarity"):
        argv(slot_similarity=1.5)
    with pytest.raises(ValueError, match="slot_save_path"):
        argv(slot_save_path="")
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--cache-ram", "24576", "--slot-similarity", "0.3",
                   "--slot-save-path", "/tmp/slots", "--dry-run"])
    assert rc == 0 and "--cache-ram 24576" in out and "--slot-prompt-similarity 0.3" in out
    assert "--slot-save-path /tmp/slots" in out


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


def test_matmul_routing_is_environment_not_argv():
    base = {"PATH": "/usr/bin"}
    assert "GGML_CUDA_NO_MMVQ" not in llama_server_env(base=base)
    assert llama_server_env(no_mmvq=True, base=base)["GGML_CUDA_NO_MMVQ"] == "1"
    assert "GGML_CUDA_NO_MMVQ" not in " ".join(argv())
    assert llama_server_command(MODEL, no_mmvq=True).startswith("GGML_CUDA_NO_MMVQ=1 ")
    assert not llama_server_command(MODEL).startswith("GGML_CUDA")
    # the threshold: keep batch-1 on the vector path, route verification to MMQ
    assert env_prefix(mmvq_max=1) == {"GGML_CUDA_MMVQ_MAX": "1"}
    assert env_prefix(no_mmvq=True, mmvq_max=1) == {"GGML_CUDA_NO_MMVQ": "1"}  # no_mmvq wins
    assert llama_server_command(MODEL, mmvq_max=1).startswith("GGML_CUDA_MMVQ_MAX=1 ")
    # the cap is ggml's MMVQ_MAX_BATCH_SIZE, raised to 32 in this fork: 32 is the widest
    # batch with a kernel instantiation, and anything past it has no `case` to dispatch to.
    assert env_prefix(mmvq_max=32) == {"GGML_CUDA_MMVQ_MAX": "32"}
    with pytest.raises(ValueError, match="mmvq_max"):
        env_prefix(mmvq_max=33)


def test_device_pin_is_environment_for_the_3060_side_server():
    # Hermes children and the compaction summary go to a ~9B model on the RTX
    # 3060 so they stop occupying 170HX slots; pinning is CUDA_VISIBLE_DEVICES,
    # never a different --main-gpu (it stays 0 inside the pinned process).
    assert env_prefix(device=1) == {"CUDA_VISIBLE_DEVICES": "1"}
    assert env_prefix(device=1, mmvq_max=3) == {
        "CUDA_VISIBLE_DEVICES": "1", "GGML_CUDA_MMVQ_MAX": "3"}
    assert "CUDA_VISIBLE_DEVICES" not in env_prefix()
    with pytest.raises(ValueError, match="device"):
        env_prefix(device=-1)
    cmd = llama_server_command(MODEL, device=1, port=8081, alias="side")
    assert cmd.startswith("CUDA_VISIBLE_DEVICES=1 ") and "--alias side" in cmd
    assert "--main-gpu 0" in cmd
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--device", "1", "--port", "8081", "--dry-run"])
    assert rc == 0 and "CUDA_VISIBLE_DEVICES=1 " in out and "--port 8081" in out


def test_cli_serve_llamacpp_dry_run_prints_the_launch():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL, "--dry-run"])
    assert rc == 0
    assert "llama-server" in out and "--kv-unified" in out and "-np 4" in out
    assert "--spec-type" not in out
    assert "dry run" in out


def test_cli_mmvq_max_flag():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mmvq-max", "1", "--dry-run"])
    assert rc == 0 and out.startswith("$ GGML_CUDA_MMVQ_MAX=1 ")
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mmvq-max", "32", "--dry-run"])
    assert rc == 0 and out.startswith("$ GGML_CUDA_MMVQ_MAX=32 ")
    rc, _ = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                 "--mmvq-max", "33", "--dry-run"])
    assert rc == 2


def test_cli_mtp_draft_flag():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mtp-draft", "1", "--dry-run"])
    assert rc == 0 and "--spec-type draft-mtp --spec-draft-n-max 1" in out
    assert "--temp 0 --repeat-penalty 1.0" in out


def test_cli_sampling_flag():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--sampling", "client", "--dry-run"])
    assert rc == 0 and "--temp" not in out


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


def test_ngram_drafting_is_off_by_default_and_ordered_before_the_mtp_head():
    # The MTP depth sweep measured the head as shallow, not broken: n-max 1 is
    # +35%, n-max 3 is -15%, n-max 5 is -22%. So width cannot come from a deeper
    # MTP draft. ngram-mod is a different source - it replays a run already in
    # the context and runs no model - so it can be deep without paying for
    # columns that will not be accepted.
    assert not any(x.startswith("--spec-ngram") for x in argv())

    mtp = "/models/Qwen3.8-27B-TurboFCFusion-MTP-Q4_K_M.gguf"
    a = build_llama_server_argv(mtp, ngram_draft=8)
    # order is load-bearing: common_speculative takes the FIRST implementation
    # that returns a non-empty draft and never concatenates, so ngram-mod must
    # precede draft-mtp or the always-drafting MTP head silences it entirely
    assert after(a, "--spec-type") == "ngram-mod,draft-mtp"
    assert after(a, "--spec-ngram-mod-n-max") == "8"
    assert after(a, "--spec-draft-n-max") == "1"     # the measured optimum, untouched

    # ngram-mod alone, on a GGUF with no MTP head
    b = argv(ngram_draft=4)
    assert after(b, "--spec-type") == "ngram-mod"
    assert not any(x.startswith("--spec-draft") for x in b)


def test_ngram_min_defaults_below_the_depth_because_llama_cpps_own_default_drafts_nothing():
    # llama.cpp defaults n_min 48 against n_max 64. ngram-mod discards the whole
    # draft when the replay ends before n_min, so inheriting 48 under a depth-8
    # draft would silently never draft: every request falls through to the MTP
    # path and the only symptom is a speedup that never arrives.
    assert after(argv(ngram_draft=8), "--spec-ngram-mod-n-min") == "4"
    assert after(argv(ngram_draft=2), "--spec-ngram-mod-n-min") == "2"   # clamped to the depth
    assert after(argv(ngram_draft=8, ngram_min=1), "--spec-ngram-mod-n-min") == "1"
    with pytest.raises(ValueError, match="ngram_min"):
        argv(ngram_draft=8, ngram_min=48)
    with pytest.raises(ValueError, match="ngram_min"):
        argv(ngram_draft=8, ngram_min=0)
    with pytest.raises(ValueError, match="ngram_draft is None"):
        argv(ngram_min=4)


def test_ngram_depth_is_bounded_by_the_flat_mmq_region():
    # The width sweep measured MMQ flat at ~55 ms from N=2 to N=16, which is why
    # draft columns inside that region are close to free. Past it each column is
    # paid, so a depth that pushes 1 + n_max over 16 is refused rather than
    # quietly turning a speed knob into a slowdown.
    assert after(argv(ngram_draft=15), "--spec-ngram-mod-n-max") == "15"
    with pytest.raises(ValueError, match="ngram_draft"):
        argv(ngram_draft=16)
    with pytest.raises(ValueError, match="ngram_draft"):
        argv(ngram_draft=0)
    with pytest.raises(ValueError, match="ngram_match"):
        argv(ngram_draft=8, ngram_match=0)
    assert after(argv(ngram_draft=8), "--spec-ngram-mod-n-match") == "24"


def test_cli_ngram_flags_reach_the_launch():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--ngram-draft", "8", "--ngram-min", "2", "--dry-run"])
    out = buf.getvalue()
    assert rc == 0
    assert "--spec-type ngram-mod" in out
    assert "--spec-ngram-mod-n-max 8 --spec-ngram-mod-n-min 2" in out


def test_backend_sampling_is_opt_in_and_emits_bs():
    # The logit row is vocab_size floats (248,320 on this model, ~0.99 MB) and this box's
    # link is PCIe Gen2 x4. Sampling on the GPU skips that copy - but it is off by default
    # in llama.cpp and must be asked for, so the launch has to emit the flag explicitly.
    assert "-bs" not in build_llama_server_argv(MODEL)
    assert "-bs" in build_llama_server_argv(MODEL, backend_sampling=True)


def test_cli_backend_sampling_flag():
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--backend-sampling", "--dry-run"])
    assert rc == 0 and " -bs " in out
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL, "--dry-run"])
    assert rc == 0 and " -bs " not in out


def test_extra_appends_raw_llama_server_args():
    # Diagnostic escape hatch: -v turns on ggml's debug log, which is the only way to see
    # "CUDA graph warmup reset". Absent unless asked for, so production launches are unchanged.
    assert "-v" not in build_llama_server_argv(MODEL)
    a = build_llama_server_argv(MODEL, extra=["-v"])
    assert a[-1] == "-v"
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL, "--extra=-v", "--dry-run"])
    assert rc == 0 and " -v\n" in out          # last flag on the command line


def test_synth_len_is_bounded_by_the_draft_depth():
    # --spec-synth-len is an instrument: it accepts draft tokens at the rate whose mean
    # accepted length is L, with the drafter bypassed. A step that drafts d tokens can
    # accept at most d + 1 including the target's own token, so L past that is meaningless
    # and llama.cpp would reject it at startup - catch it in the launch builder instead.
    a = build_llama_server_argv(MODEL, mtp_draft=3, synth_len=2.5)
    assert after(a, "--spec-synth-len") == "2.5"
    assert "--spec-synth-len" not in build_llama_server_argv(MODEL, mtp_draft=3)
    with pytest.raises(ValueError, match="synth_len"):
        build_llama_server_argv(MODEL, mtp_draft=1, synth_len=9)
    with pytest.raises(ValueError, match="synth_len"):
        build_llama_server_argv(MODEL, mtp_draft=1, synth_len=0.5)
    rc, out = run(["serve", "--backend", "llamacpp", "--model", MODEL,
                   "--mtp-draft", "3", "--synth-len", "2", "--dry-run"])
    assert rc == 0 and "--spec-synth-len 2" in out
