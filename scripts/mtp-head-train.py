#!/usr/bin/env python3
"""
mtp-head-train - re-align the Qwen3.5/3.8 MTP ("nextn") draft head to a merged
trunk, Route A of docs/mtp-realign-davidau.md.

The trunk (the served merge, HF safetensors) is loaded frozen - optionally in
4-bit NF4 so the head learns against statistics close to the Q4_K_M that is
actually served - and run once per batch under no_grad. Only the MTP head is
trained: hnorm (RMSNorm on the trunk's post-final-norm hidden state h_t),
enorm (RMSNorm on the shared token embedding of token t+1), fc / eh_proj
(concat 2h -> h), one full-attention decoder layer built from the model's own
layer class (same hidden size, heads, KV heads, gated attention, MRoPE), and
the shared_head norm. embed_tokens and lm_head are the trunk's and stay frozen.

Objective: cross-entropy of the head's prediction at position t against the
token at t+2 (the trunk predicts t+1; the head predicts one further). With
--depth K > 1 the head is also unrolled the way llama.cpp drafts: step k feeds
the head its own post-shared_head_norm hidden state from step k-1 plus the
embedding of the TRUE token t+k (teacher forcing) and predicts token t+k+1;
the step losses are combined with --depth-weights.

Wiring follows this repo's graph exactly (src/models/qwen35.cpp, graph_mtp):
    h_t    = trunk output after `output_norm`       (h_nextn)
    x      = eh_proj([enorm(embed(tok_{t+1})) ; hnorm(h_t)])
    x      = full-attention decoder block(x)
    hk     = shared_head_norm(x)                     (fed back for step k+1)
    logits = lm_head(hk)                             -> target tok_{t+2}

Data: JSONL from scripts/mtp-realign-gen.py ({"messages", "reasoning",
"completion", ...}). Each row is rendered with the model's chat template with
the thinking kept inline, tokenized, and packed into --seq-len windows. Nothing
is masked: every position trains (the head must draft prompts, tool output and
thinking alike, because the server drafts across all of it).

Init: --init-from <hf dir | .safetensors> loads the stock base-model head
(`mtp.*` tensors); without it the merge's own `mtp.*` tensors are used.

Output (--out-dir):
    mtp-head.safetensors    the head in HF names: mtp.fc.weight,
                            mtp.pre_fc_norm_embedding.weight,
                            mtp.pre_fc_norm_hidden.weight, mtp.norm.weight,
                            mtp.layers.0.{self_attn,mlp,*_layernorm}.*
                            - exactly what convert_hf_to_gguf.py --mtp remaps
                            (conversion/qwen.py _Qwen35MtpMixin) to
                            blk.N.nextn.{eh_proj,enorm,hnorm,shared_head_norm}
                            and blk.N.{attn_*,ffn_*} in the GGUF.
    mtp-head.json           config, training stats, the tensor-name map.
    ckpt-latest/            resumable checkpoint (--resume).
--install-dir DIR then builds an export directory: symlinks to the trunk
shards/tokenizer/config, the old shard(s) rewritten without the stock `mtp.*`,
a new shard with the trained head, and a consistent model.safetensors.index.json
so `convert_hf_to_gguf.py --mtp --outtype q8_0 DIR` produces the sidecar.

Eval (--eval): per depth position k = 1..K, top-1 agreement of the head's
prediction with the trunk's argmax for the same target position (the
acceptance proxy the doc names: greedy acceptance is exact match against the
target's argmax), plus top-1 accuracy against the data token, on --eval-data.

VRAM (27B trunk, hidden 5120, vocab ~248k, --seq-len 2048, --batch-size 1):
    trunk NF4 (--trunk-4bit)          ~15-16 GiB   (bf16 would be ~54 GiB: does not fit)
    head fp32 master + grads + AdamW  ~6.5 GiB     (~0.4B params x 16 bytes)
    trunk activations, no_grad        ~2-3 GiB
    head activations + chunked logits ~3-4 GiB     (logits are computed in
                                                    --logit-chunk slices with
                                                    recompute, never T x vocab at once)
    total                             ~28-30 GiB   -> fits the 40 GiB card; use
                                                    --batch-size 2 or --seq-len 4096
                                                    only with --grad-accum kept >= 4.

Dependencies: torch, transformers, safetensors (bitsandbytes for --trunk-4bit).
All imported lazily. `--self-test` needs NONE of them: it exercises the
argument parsing, the JSONL loader / chat rendering / packing (with a stub
tokenizer), the depth-alignment index arithmetic, the tensor-name mapping and
the config/JSON plumbing, and exits 0 on a machine without torch.

  python3 scripts/mtp-head-train.py --model /models/Qwen3.8-27B-TURBO-NM-DAU \\
      --data gen.jsonl --init-from /models/Qwen3.8-27B --trunk-4bit \\
      --depth 3 --seq-len 2048 --grad-accum 8 --epochs 2 --out-dir mtp-retrain
  python3 scripts/mtp-head-train.py --model ... --head mtp-retrain/mtp-head.safetensors \\
      --eval --eval-data heldout.jsonl --trunk-4bit --depth 3
  python3 scripts/mtp-head-train.py --model ... --head mtp-retrain/mtp-head.safetensors \\
      --install-dir /models/Qwen3.8-27B-TURBO-NM-DAU-mtp-retrained
  python3 scripts/mtp-head-train.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# tensor names: HF checkpoint (`mtp.*`)  <->  this script's module attributes
#                                        --> GGUF via convert_hf_to_gguf.py --mtp
# --------------------------------------------------------------------------

# HF name in the checkpoint -> attribute path inside MTPHead (state_dict key).
HEAD_TOP_LEVEL = {
    "mtp.fc.weight":                      "fc.weight",
    "mtp.pre_fc_norm_embedding.weight":   "enorm.weight",
    "mtp.pre_fc_norm_hidden.weight":      "hnorm.weight",
    "mtp.norm.weight":                    "shared_head_norm.weight",
}
HEAD_LAYER_PREFIX_HF = "mtp.layers.0."      # decoder block tensors
HEAD_LAYER_PREFIX_LOCAL = "layer."

# What convert_hf_to_gguf.py --mtp turns each HF name into (for the JSON and the
# docstring; the exporter itself does the remap - conversion/qwen.py).
GGUF_NAME_HINT = {
    "mtp.fc.weight":                    "blk.{L}.nextn.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "blk.{L}.nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight":    "blk.{L}.nextn.hnorm.weight",
    "mtp.norm.weight":                  "blk.{L}.nextn.shared_head_norm.weight",
    "mtp.layers.0.input_layernorm.weight":           "blk.{L}.attn_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight":  "blk.{L}.post_attention_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight":          "blk.{L}.attn_q.weight  (query+gate, 2*heads*head_dim rows)",
    "mtp.layers.0.self_attn.k_proj.weight":          "blk.{L}.attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight":          "blk.{L}.attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight":          "blk.{L}.attn_output.weight",
    "mtp.layers.0.self_attn.q_norm.weight":          "blk.{L}.attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight":          "blk.{L}.attn_k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight":             "blk.{L}.ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight":               "blk.{L}.ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight":             "blk.{L}.ffn_down.weight",
}
# "{L}" is num_hidden_layers of the trunk (64 for the 27B): the exporter appends
# the MTP block after the last trunk block.

HEAD_SHARD_NAME = "model-mtp-retrained.safetensors"


def hf_to_local(name: str) -> str | None:
    """Map an HF checkpoint tensor name to the MTPHead state_dict key (None = not a head tensor)."""
    if name.startswith("model."):
        name = name[len("model."):]
    if name in HEAD_TOP_LEVEL:
        return HEAD_TOP_LEVEL[name]
    if name.startswith(HEAD_LAYER_PREFIX_HF):
        return HEAD_LAYER_PREFIX_LOCAL + name[len(HEAD_LAYER_PREFIX_HF):]
    return None


def local_to_hf(key: str) -> str:
    """Inverse of hf_to_local for saving."""
    for hf, local in HEAD_TOP_LEVEL.items():
        if key == local:
            return hf
    if key.startswith(HEAD_LAYER_PREFIX_LOCAL):
        return HEAD_LAYER_PREFIX_HF + key[len(HEAD_LAYER_PREFIX_LOCAL):]
    raise KeyError(f"not a head tensor: {key}")


# --------------------------------------------------------------------------
# config plumbing (no torch)
# --------------------------------------------------------------------------

def load_hf_config(model_dir: Path) -> dict:
    """config.json, flattened: ForConditionalGeneration nests the text config."""
    cfg = json.loads((model_dir / "config.json").read_text())
    text = cfg.get("text_config") or {}
    flat = {**cfg, **text}
    return flat


def full_attention_layer_index(cfg: dict) -> int:
    """Index of a full-attention layer in the trunk; the MTP block copies its structure."""
    types = cfg.get("layer_types")
    if types:
        for i, t in enumerate(types):
            if t == "full_attention":
                return i
    interval = int(cfg.get("full_attention_interval", 4))
    return interval - 1


def head_config(cfg: dict) -> dict:
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    return {
        "hidden_size": hidden,
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_attention_heads": heads,
        "num_key_value_heads": int(cfg.get("num_key_value_heads", heads)),
        "head_dim": int(cfg.get("head_dim") or hidden // heads),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "vocab_size": int(cfg["vocab_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "mtp_num_hidden_layers": int(cfg.get("mtp_num_hidden_layers", 1)),
        "full_attention_layer_index": full_attention_layer_index(cfg),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
    }


def head_param_count(hc: dict) -> int:
    h, ff = hc["hidden_size"], hc["intermediate_size"]
    nh, nkv, hd = hc["num_attention_heads"], hc["num_key_value_heads"], hc["head_dim"]
    attn = h * (2 * nh * hd) + 2 * h * (nkv * hd) + (nh * hd) * h + 2 * hd
    mlp = 3 * h * ff
    return attn + mlp + 2 * h * h + 6 * h


def depth_weights(depth: int, spec: str | None) -> list[float]:
    if spec:
        ws = [float(x) for x in spec.split(",")]
        if len(ws) != depth:
            raise ValueError(f"--depth-weights needs {depth} values, got {len(ws)}")
        return ws
    return [0.8 ** k for k in range(depth)]


# --------------------------------------------------------------------------
# data (no torch): JSONL -> chat text -> token ids -> packed windows
# --------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_conversation(row: dict) -> list[dict]:
    """Prompt messages + the regenerated assistant turn with its thinking."""
    msgs = list(row.get("messages") or [{"role": "user", "content": row.get("prompt", "")}])
    reasoning = (row.get("reasoning") or "").strip()
    completion = row.get("completion") or ""
    turn = {"role": "assistant", "content": completion}
    if reasoning:
        turn["reasoning_content"] = reasoning
    msgs.append(turn)
    return msgs


def render_chat(tokenizer, conv: list[dict]) -> str:
    """Apply the chat template; make sure the thinking survived (some templates
    drop reasoning_content of earlier turns - ours is the final turn, but be safe)."""
    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    last = conv[-1]
    reasoning = last.get("reasoning_content")
    if reasoning and reasoning[:40] not in text:
        # template ignored reasoning_content: inline it the way the model emits it
        conv2 = conv[:-1] + [{"role": "assistant",
                              "content": f"<think>\n{reasoning}\n</think>\n\n{last['content']}"}]
        text = tokenizer.apply_chat_template(conv2, tokenize=False, add_generation_prompt=False)
    return text


def tokenize_rows(tokenizer, rows: list[dict], *, log_every: int = 2000) -> list[list[int]]:
    docs = []
    for i, row in enumerate(rows):
        text = render_chat(tokenizer, row_to_conversation(row))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 3:
            docs.append(ids)
        if log_every and (i + 1) % log_every == 0:
            print(f"  tokenized {i + 1}/{len(rows)}", file=sys.stderr)
    return docs


def pack(docs: list[list[int]], seq_len: int, *, shuffle: bool, seed: int = 0) -> list[list[int]]:
    """Concatenate documents and cut into fixed windows. Window length is
    seq_len + 1 so a window of T tokens carries targets up to t+K (the
    alignment code slices per depth). No masking: every position trains."""
    order = list(range(len(docs)))
    if shuffle:
        random.Random(seed).shuffle(order)
    stream: list[int] = []
    for i in order:
        stream.extend(docs[i])
    win = seq_len + 1
    n = len(stream) // win
    return [stream[i * win:(i + 1) * win] for i in range(n)]


def depth_slices(T: int, k: int) -> tuple[slice, slice, slice, int]:
    """For depth step k (1-based) on a window of T tokens:
    returns (hidden rows, embedded-token cols, target cols, valid length L).
    Step 1: hidden h[:, :T-2], tokens x[:, 1:T-1], target x[:, 2:T].
    Step k: prev head hidden [:, :L], tokens x[:, k:k+L], target x[:, k+1:k+1+L]."""
    L = T - k - 1
    if L <= 0:
        raise ValueError(f"window of {T} tokens too short for depth {k}")
    return slice(0, L), slice(k, k + L), slice(k + 1, k + 1 + L), L


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--model", help="HF directory of the merged trunk (bf16 safetensors + tokenizer)")
    g.add_argument("--trunk-4bit", action="store_true",
                   help="load the trunk in NF4 via bitsandbytes (matches the served Q4_K_M statistics; needed on 40 GiB)")
    g.add_argument("--trunk-8bit", action="store_true", help="load the trunk in int8 via bitsandbytes")
    g.add_argument("--init-from", help="HF dir or .safetensors with stock `mtp.*` weights (base model = healthier start); default: the merge's own")
    g.add_argument("--head", help="trained head .safetensors (for --eval / --install-dir, or to continue training)")
    g.add_argument("--attn-impl", default="sdpa", help="transformers attn_implementation for trunk and head (default sdpa)")
    g.add_argument("--device", default="cuda")

    g = ap.add_argument_group("data")
    g.add_argument("--data", help="JSONL from scripts/mtp-realign-gen.py")
    g.add_argument("--eval-data", help="held-out JSONL for --eval / periodic eval (default: last --eval-frac of --data)")
    g.add_argument("--eval-frac", type=float, default=0.02)
    g.add_argument("--seq-len", type=int, default=2048)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--max-windows", type=int, default=0, help="cap the number of training windows (0 = all)")
    g.add_argument("--seed", type=int, default=1234)

    g = ap.add_argument_group("objective")
    g.add_argument("--depth", type=int, default=3, help="K: also train recursive steps 2..K (teacher-forced)")
    g.add_argument("--depth-weights", help="comma list of K loss weights (default 0.8^(k-1))")
    g.add_argument("--logit-chunk", type=int, default=512, help="positions per lm_head slice (memory)")

    g = ap.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--min-lr-ratio", type=float, default=0.1)
    g.add_argument("--warmup-steps", type=int, default=50)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--grad-accum", type=int, default=8)
    g.add_argument("--epochs", type=float, default=2.0)
    g.add_argument("--max-steps", type=int, default=0, help="optimizer steps cap (0 = from epochs)")

    g = ap.add_argument_group("output")
    g.add_argument("--out-dir", default="mtp-retrain")
    g.add_argument("--save-every", type=int, default=100, help="checkpoint every N optimizer steps")
    g.add_argument("--eval-every", type=int, default=200, help="run the agreement eval every N optimizer steps (0 = only at the end)")
    g.add_argument("--eval-windows", type=int, default=32, help="windows per periodic eval")
    g.add_argument("--metric-every", type=int, default=10, help="compute the train-time agreement metric every N micro-batches")
    g.add_argument("--resume", action="store_true", help="resume from <out-dir>/ckpt-latest")
    g.add_argument("--install-dir", help="build an export directory for convert_hf_to_gguf.py --mtp from --model and --head")

    g = ap.add_argument_group("modes")
    g.add_argument("--eval", action="store_true", help="evaluate --head (or the stock head) and exit")
    g.add_argument("--self-test", action="store_true", help="exercise the torch-free plumbing and exit")
    return ap


# --------------------------------------------------------------------------
# torch side
# --------------------------------------------------------------------------

def _import_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        sys.exit(f"missing dependency: {e}. Install torch, transformers, safetensors "
                 "(and bitsandbytes for --trunk-4bit/--trunk-8bit).")


def load_trunk(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = dict(torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={"": 0} if args.device == "cuda" else args.device)
    if args.trunk_4bit or args.trunk_8bit:
        from transformers import BitsAndBytesConfig
        if args.trunk_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                # keep the parts the head shares with the trunk in bf16
                llm_int8_skip_modules=["lm_head", "embed_tokens", "norm"])
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                           llm_int8_skip_modules=["lm_head", "embed_tokens", "norm"])
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.model
    if hasattr(inner, "language_model"):     # ForConditionalGeneration wrapper
        inner = inner.language_model
    return tok, model, inner


class TrunkTap:
    """Forward hook on the trunk's final norm: captures its output (h_nextn)."""
    def __init__(self, norm_module):
        self.out = None
        self.h = norm_module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.out = out

    def remove(self):
        self.h.remove()


def build_head(inner, cfg: dict, hc: dict):
    """MTPHead assembled from the model's own classes so structure matches."""
    import torch
    import torch.nn as nn

    layer_idx = hc["full_attention_layer_index"]
    ref_layer = inner.layers[layer_idx]
    norm_cls = type(inner.norm)
    model_cfg = inner.config
    h = hc["hidden_size"]

    class MTPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.enorm = norm_cls(h, eps=hc["rms_norm_eps"])
            self.hnorm = norm_cls(h, eps=hc["rms_norm_eps"])
            self.fc = nn.Linear(2 * h, h, bias=False)
            self.layer = type(ref_layer)(model_cfg, layer_idx)
            self.shared_head_norm = norm_cls(h, eps=hc["rms_norm_eps"])

        def forward(self, h_prev, tok_embd, position_ids, attn_mask, rotary):
            # llama.cpp concatenates [enorm ; hnorm] (dim 0 in ggml = features), eh_proj on that
            x = self.fc(torch.cat([self.enorm(tok_embd), self.hnorm(h_prev)], dim=-1))
            pe = _rotary(rotary, x, position_ids)
            out = self.layer(x, attention_mask=attn_mask, position_ids=position_ids[-1] if position_ids.dim() == 3 else position_ids,
                             position_embeddings=pe, use_cache=False)
            if isinstance(out, tuple):
                out = out[0]
            return self.shared_head_norm(out)

    head = MTPHead()
    return head


def _rotary(rotary, x, position_ids):
    """Qwen3.5 uses MRoPE with (3, B, T) positions; older layouts take (B, T)."""
    try:
        return rotary(x, position_ids)
    except (RuntimeError, IndexError, ValueError):
        return rotary(x, position_ids[0] if position_ids.dim() == 3 else position_ids[None].expand(3, *position_ids.shape))


def make_positions(B: int, L: int, offset: int, device):
    import torch
    pos = torch.arange(offset, offset + L, device=device)[None].expand(B, L)
    return pos[None].expand(3, B, L).contiguous()   # MRoPE layout; _rotary falls back if 2D is wanted


def causal_mask(B: int, L: int, dtype, device):
    import torch
    m = torch.full((L, L), torch.finfo(dtype).min, dtype=dtype, device=device).triu_(1)
    return m[None, None].expand(B, 1, L, L)


def load_head_weights(head, src: Path, *, strict: bool = True) -> int:
    """Load `mtp.*` tensors from an HF dir (via its index / shards) or a single safetensors file."""
    from safetensors import safe_open
    import torch

    files: list[Path]
    if src.is_dir():
        idx = src / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            files = sorted({src / v for k, v in wm.items() if hf_to_local(k)})
        else:
            files = sorted(src.glob("*.safetensors"))
    else:
        files = [src]
    sd = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for name in st.keys():
                local = hf_to_local(name)
                if local:
                    sd[local] = st.get_tensor(name)
    if not sd:
        raise FileNotFoundError(f"no `mtp.*` tensors found in {src}")
    missing, unexpected = head.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()}, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"head init mismatch: missing={missing} unexpected={unexpected}")
    return len(sd)


def save_head(head, path: Path, meta: dict):
    from safetensors.torch import save_file
    import torch
    sd = {local_to_hf(k): v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in head.state_dict().items()}
    save_file(sd, str(path), metadata={"format": "pt", **{k: str(v) for k, v in meta.items()}})


def chunked_logits_ce(lm_head, hidden, targets, chunk: int, trunk_argmax=None):
    """Sum CE over positions in slices (recompute in backward), plus top-1 stats.
    hidden (N, h) float, targets (N,) long. Returns (loss_sum, correct, agree, N)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def ce_slice(hs, ts):
        logits = lm_head(hs).float()
        return torch.nn.functional.cross_entropy(logits, ts, reduction="sum"), logits.argmax(-1)

    total = hidden.new_zeros((), dtype=torch.float32)
    correct = 0
    agree = 0
    N = hidden.shape[0]
    for s in range(0, N, chunk):
        hs, ts = hidden[s:s + chunk], targets[s:s + chunk]
        if hidden.requires_grad:
            loss, pred = checkpoint(ce_slice, hs, ts, use_reentrant=False)
        else:
            loss, pred = ce_slice(hs, ts)
        total = total + loss
        correct += int((pred == ts).sum())
        if trunk_argmax is not None:
            agree += int((pred == trunk_argmax[s:s + chunk]).sum())
    return total, correct, agree, N


def trunk_argmax_all(lm_head, h, chunk: int):
    """Trunk's greedy token at every position (what the server would accept against)."""
    import torch
    with torch.no_grad():
        B, T, _ = h.shape
        flat = h.reshape(B * T, -1)
        out = torch.empty(B * T, dtype=torch.long, device=h.device)
        for s in range(0, B * T, chunk):
            out[s:s + chunk] = lm_head(flat[s:s + chunk]).argmax(-1)
        return out.view(B, T)


def run_head_on_window(head, inner, lm_head, x, h_trunk, *, depth: int, weights: list[float],
                       logit_chunk: int, want_metric: bool, rotary):
    """One packed window batch. x: (B, T) tokens, h_trunk: (B, T, h) post-norm trunk hidden.
    Returns (weighted loss, per-depth dict of stats)."""
    import torch
    B, T = x.shape
    embd = inner.embed_tokens
    dtype = h_trunk.dtype
    trunk_pred = trunk_argmax_all(lm_head, h_trunk, logit_chunk) if want_metric else None
    stats = {}
    total = h_trunk.new_zeros((), dtype=torch.float32)
    prev = h_trunk
    for k in range(1, depth + 1):
        hs, tokc, tgtc, L = depth_slices(T, k)
        h_in = prev[:, hs]
        e_in = embd(x[:, tokc]).to(dtype)
        pos = make_positions(B, L, k - 1, x.device)
        mask = causal_mask(B, L, dtype, x.device)
        hk = head(h_in, e_in, pos, mask, rotary)          # (B, L, h) post shared_head_norm
        tgt = x[:, tgtc].reshape(-1)
        # trunk's own prediction for the same target token sits one row earlier:
        # trunk at position p predicts token p+1; target index = t+k+1 -> trunk row t+k
        ta = trunk_pred[:, k:k + L].reshape(-1) if want_metric else None
        loss_sum, correct, agree, N = chunked_logits_ce(lm_head, hk.reshape(B * L, -1), tgt, logit_chunk, ta)
        loss_k = loss_sum / N
        total = total + weights[k - 1] * loss_k
        stats[k] = {"loss": float(loss_k), "acc": correct / N, "agree": (agree / N) if want_metric else None}
        prev = hk
    return total, stats


def fmt_stats(stats: dict) -> str:
    parts = []
    for k, s in stats.items():
        a = f" agree {s['agree']:.3f}" if s["agree"] is not None else ""
        parts.append(f"k{k}: loss {s['loss']:.3f} acc {s['acc']:.3f}{a}")
    return " | ".join(parts)


def evaluate(head, inner, lm_head, tap, windows, args, weights, rotary, *, limit: int = 0) -> dict:
    import torch
    head.eval()
    agg: dict[int, dict[str, float]] = {}
    n = 0
    dev = args.device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for bi in range(0, len(windows), args.batch_size):
            if limit and n >= limit:
                break
            batch = windows[bi:bi + args.batch_size]
            x = torch.tensor(batch, dtype=torch.long, device=dev)
            inner(input_ids=x, use_cache=False)
            _, stats = run_head_on_window(head, inner, lm_head, x, tap.out, depth=args.depth, weights=weights,
                                          logit_chunk=args.logit_chunk, want_metric=True, rotary=rotary)
            for k, s in stats.items():
                a = agg.setdefault(k, {"loss": 0.0, "acc": 0.0, "agree": 0.0})
                for key in a:
                    a[key] += s[key]
            n += 1
    head.train()
    out = {k: {key: v / max(n, 1) for key, v in s.items()} for k, s in agg.items()}
    return out


def print_eval(res: dict, title: str):
    print(f"[{title}] position-wise (k = draft position; agree = top-1 match with trunk argmax = greedy acceptance proxy)")
    for k, s in res.items():
        print(f"  position {k}: agree {s['agree']:.3f}  acc {s['acc']:.3f}  loss {s['loss']:.3f}")
    if res:
        # expected accepted tokens per pass if acceptance were independent per position (rough)
        acc = 0.0
        p = 1.0
        for k in sorted(res):
            p *= res[k]["agree"]
            acc += p
        print(f"  ~accepted/pass at n-max {len(res)} (chained agree): {acc:.2f}")


def export_commands(install_dir: str, out_dir: str) -> str:
    return (
        f"# 1. install the head into an export directory (symlinks the trunk, swaps mtp.*):\n"
        f"python3 scripts/mtp-head-train.py --model <HF trunk dir> --head {out_dir}/mtp-head.safetensors --install-dir {install_dir}\n"
        f"# 2. export just the head as the mtp-*.gguf sidecar (Q8_0, as DavidAU ships):\n"
        f"python3 convert_hf_to_gguf.py --mtp --outtype q8_0 {install_dir} --outfile {out_dir}/\n"
        f"# 3. serve (--mtp-draft 3 == --spec-draft-n-max 3; the sidecar needs -md, which\n"
        f"#    `tenselerate serve` does not pass through yet - take its --dry-run argv and add):\n"
        f"#    GGML_CUDA_NO_MMVQ=1 llama-server -m <trunk>.gguf -md {out_dir}/mtp-*.gguf \\\n"
        f"#        --spec-type draft-mtp --spec-draft-n-max 3 --temp 0 --repeat-penalty 1.0\n"
    )


def train(args) -> int:
    _import_torch()
    import torch

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model)
    cfg = load_hf_config(model_dir)
    hc = head_config(cfg)
    weights = depth_weights(args.depth, args.depth_weights)
    print(f"head config: {json.dumps(hc)}\n~{head_param_count(hc) / 1e9:.2f}B trainable params, depth {args.depth}, weights {weights}")

    print("loading trunk ...", file=sys.stderr)
    tok, model, inner = load_trunk(args)
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    rotary = inner.rotary_emb
    tap = TrunkTap(inner.norm)

    head = build_head(inner, cfg, hc).to(args.device).float()
    init_src = Path(args.head) if args.head else Path(args.init_from) if args.init_from else model_dir
    n_loaded = load_head_weights(head, init_src)
    print(f"head initialised from {init_src} ({n_loaded} tensors)")

    # data
    print("tokenizing ...", file=sys.stderr)
    rows = read_rows(Path(args.data)) if args.data else []
    eval_rows = read_rows(Path(args.eval_data)) if args.eval_data else []
    if not eval_rows and rows and not args.eval:
        n_eval = max(1, int(len(rows) * args.eval_frac))
        rows, eval_rows = rows[:-n_eval], rows[-n_eval:]
    if args.eval and not eval_rows:
        eval_rows = rows
        rows = []
    train_windows = pack(tokenize_rows(tok, rows), args.seq_len, shuffle=True, seed=args.seed) if rows else []
    eval_windows = pack(tokenize_rows(tok, eval_rows), args.seq_len, shuffle=False) if eval_rows else []
    if args.max_windows:
        train_windows = train_windows[:args.max_windows]
    print(f"{len(train_windows)} train windows, {len(eval_windows)} eval windows of {args.seq_len} tokens")

    if args.eval:
        res = evaluate(head, inner, lm_head, tap, eval_windows, args, weights, rotary)
        print_eval(res, f"eval {init_src}")
        (out_dir / "eval.json").write_text(json.dumps(res, indent=1))
        return 0

    # optimiser
    micro_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    steps_per_epoch = max(1, micro_per_epoch // args.grad_accum)
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    step = 0
    micro = 0
    ckpt = out_dir / "ckpt-latest"
    if args.resume and (ckpt / "state.pt").exists():
        state = torch.load(ckpt / "state.pt", map_location="cpu")
        load_head_weights(head, ckpt / "head.safetensors")
        opt.load_state_dict(state["opt"])
        step, micro = state["step"], state["micro"]
        print(f"resumed at step {step} (micro {micro})")

    log = (out_dir / "train.log").open("a")
    head.train()
    t0 = time.time()
    dev = args.device
    while step < total_steps:
        rng = random.Random(args.seed + micro // micro_per_epoch)
        order = list(range(len(train_windows)))
        rng.shuffle(order)
        for bi in range((micro % micro_per_epoch) * args.batch_size, len(order), args.batch_size):
            idx = order[bi:bi + args.batch_size]
            batch = [train_windows[i] for i in idx]
            x = torch.tensor(batch, dtype=torch.long, device=dev)
            want_metric = args.metric_every and micro % args.metric_every == 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                inner(input_ids=x, use_cache=False)
                h = tap.out.detach()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss, stats = run_head_on_window(head, inner, lm_head, x, h, depth=args.depth, weights=weights,
                                                 logit_chunk=args.logit_chunk, want_metric=want_metric, rotary=rotary)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                el = time.time() - t0
                line = (f"step {step}/{total_steps} lr {lr_at(step - 1):.2e} loss {float(loss):.4f} gnorm {float(gn):.2f} "
                        f"| {fmt_stats(stats)} | {el / 60:.1f} min")
                print(line)
                log.write(json.dumps({"step": step, "loss": float(loss), "lr": lr_at(step - 1), "stats": stats, "t": el}) + "\n")
                log.flush()
                if args.save_every and step % args.save_every == 0:
                    ckpt.mkdir(exist_ok=True)
                    save_head(head, ckpt / "head.safetensors", {"step": step})
                    torch.save({"opt": opt.state_dict(), "step": step, "micro": micro}, ckpt / "state.pt")
                if args.eval_every and eval_windows and step % args.eval_every == 0:
                    res = evaluate(head, inner, lm_head, tap, eval_windows, args, weights, rotary, limit=args.eval_windows)
                    print_eval(res, f"step {step}")
                    log.write(json.dumps({"step": step, "eval": res}) + "\n")
                if step >= total_steps:
                    break
        if micro % micro_per_epoch != 0 and step < total_steps:
            micro += micro_per_epoch - micro % micro_per_epoch   # epoch boundary for the shuffle seed

    final_eval = evaluate(head, inner, lm_head, tap, eval_windows, args, weights, rotary) if eval_windows else {}
    if final_eval:
        print_eval(final_eval, "final")
    head_path = out_dir / "mtp-head.safetensors"
    save_head(head, head_path, {"step": step, "source_model": str(model_dir)})
    meta = {
        "head_config": hc, "depth": args.depth, "depth_weights": weights, "steps": step,
        "seq_len": args.seq_len, "lr": args.lr, "init_from": str(init_src), "model": str(model_dir),
        "trunk_quant": "nf4" if args.trunk_4bit else "int8" if args.trunk_8bit else "bf16",
        "final_eval": final_eval,
        "tensor_names": {"hf": sorted(local_to_hf(k) for k in head.state_dict()),
                         "gguf_hint": {k: v.replace("{L}", str(hc["num_hidden_layers"])) for k, v in GGUF_NAME_HINT.items()}},
    }
    (out_dir / "mtp-head.json").write_text(json.dumps(meta, indent=1))
    install = args.install_dir or f"{model_dir}-mtp-retrained"
    print(f"\nsaved {head_path} and {out_dir / 'mtp-head.json'}\n")
    print(export_commands(install, str(out_dir)))
    if args.install_dir:
        install_head(model_dir, head_path, Path(args.install_dir))
    return 0


# --------------------------------------------------------------------------
# install: build the export directory for convert_hf_to_gguf.py --mtp
# --------------------------------------------------------------------------

def rewrite_index(index: dict, head_names: list[str], new_shard: str) -> tuple[dict, set[str]]:
    """Point every `mtp.*` entry at the new shard; return (index, shards that lost tensors).
    Pure function so --self-test can check it."""
    wm = dict(index["weight_map"])
    touched = set()
    for k in list(wm):
        if hf_to_local(k):
            touched.add(wm[k])
            del wm[k]
    for n in head_names:
        wm[n] = new_shard
    out = dict(index)
    out["weight_map"] = wm
    return out, touched


def install_head(model_dir: Path, head_path: Path, install_dir: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    install_dir.mkdir(parents=True, exist_ok=True)
    idx_path = model_dir / "model.safetensors.index.json"
    with safe_open(str(head_path), framework="pt", device="cpu") as st:
        head_names = list(st.keys())
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        wm = {}
        for s in shards:
            with safe_open(str(model_dir / s), framework="pt", device="cpu") as st:
                for k in st.keys():
                    wm[k] = s
        index = {"metadata": {}, "weight_map": wm}
    new_index, touched = rewrite_index(index, head_names, HEAD_SHARD_NAME)

    for f in sorted(model_dir.iterdir()):
        if f.name == idx_path.name or f.name in touched or f.name == HEAD_SHARD_NAME:
            continue
        dst = install_dir / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    for shard in sorted(touched):
        # copy the shard without its stock mtp.* tensors (the exporter reads every
        # part and checks names against the index; duplicates would be ambiguous)
        print(f"rewriting {shard} without mtp.* ...", file=sys.stderr)
        keep = {}
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as st:
            meta = st.metadata()
            for k in st.keys():
                if not hf_to_local(k):
                    keep[k] = st.get_tensor(k)
        save_file(keep, str(install_dir / shard), metadata=meta or {"format": "pt"})
    if not (install_dir / HEAD_SHARD_NAME).exists():
        os.symlink(head_path.resolve(), install_dir / HEAD_SHARD_NAME)
    (install_dir / idx_path.name).write_text(json.dumps(new_index, indent=2))
    print(f"export directory ready: {install_dir}\n"
          f"python3 convert_hf_to_gguf.py --mtp --outtype q8_0 {install_dir}")


# --------------------------------------------------------------------------
# self-test (no torch)
# --------------------------------------------------------------------------

class _StubTokenizer:
    """Byte-level stand-in: renders a minimal chat template, encodes UTF-8 bytes."""
    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False):
        out = []
        for m in conv:
            body = m["content"]
            if m.get("reasoning_content"):
                body = f"<think>\n{m['reasoning_content']}\n</think>\n\n{body}"
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


def self_test() -> int:
    import tempfile

    # argument parsing + defaults
    ap = build_parser()
    a = ap.parse_args(["--model", "m", "--data", "d.jsonl", "--depth", "3", "--trunk-4bit"])
    assert a.lr == 1e-4 and a.grad_accum == 8 and a.seq_len == 2048 and a.trunk_4bit
    assert depth_weights(3, None) == [1.0, 0.8, 0.6400000000000001]
    assert depth_weights(2, "1,0.5") == [1.0, 0.5]
    try:
        depth_weights(3, "1,0.5")
        raise AssertionError("bad --depth-weights accepted")
    except ValueError:
        pass

    # tensor-name round trip: HF -> local -> HF, and the exporter's remap targets
    for hf in GGUF_NAME_HINT:
        local = hf_to_local(hf)
        assert local and local_to_hf(local) == hf, hf
    assert hf_to_local("model.mtp.fc.weight") == "fc.weight"
    assert hf_to_local("model.layers.3.mlp.up_proj.weight") is None
    assert hf_to_local("lm_head.weight") is None

    # config plumbing on a Qwen3.5-27B-shaped config (nested text_config form)
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {
        "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 32,
        "num_key_value_heads": 4, "head_dim": 128, "rms_norm_eps": 1e-6, "vocab_size": 248320,
        "num_hidden_layers": 64, "mtp_num_hidden_layers": 1, "full_attention_interval": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 60}}
    hc = head_config({**cfg, **cfg["text_config"]})
    assert hc["full_attention_layer_index"] == 3 and hc["num_key_value_heads"] == 4
    assert 0.3e9 < head_param_count(hc) < 0.6e9, head_param_count(hc)   # doc: ~0.4B
    hint = {k: v.replace("{L}", str(hc["num_hidden_layers"])) for k, v in GGUF_NAME_HINT.items()}
    assert hint["mtp.fc.weight"] == "blk.64.nextn.eh_proj.weight"

    # depth alignment: step k at row t sees token t+k and predicts t+k+1
    T = 16
    for k in (1, 2, 3):
        hs, tokc, tgtc, L = depth_slices(T, k)
        rows = list(range(T))
        assert rows[hs] == list(range(L))
        assert rows[tokc] == [t + k for t in range(L)]
        assert rows[tgtc] == [t + k + 1 for t in range(L)]
    assert depth_slices(16, 1)[3] == 14 and depth_slices(16, 3)[3] == 12

    # data: JSONL in gen.py's format -> chat -> packed windows
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "gen.jsonl")
        with p.open("w") as f:
            for i in range(6):
                f.write(json.dumps({"id": i, "messages": [{"role": "user", "content": f"q{i} " * 20}],
                                    "reasoning": f"think {i}", "completion": f"answer {i} " * 10,
                                    "reasoning_effort": "low", "model": "stub"}) + "\n")
            f.write(json.dumps({"id": "p", "prompt": "bare prompt", "completion": "x", "reasoning": ""}) + "\n")
        rows = read_rows(p)
        assert len(rows) == 7
        conv = row_to_conversation(rows[0])
        assert conv[-1]["role"] == "assistant" and conv[-1]["reasoning_content"] == "think 0"
        assert "reasoning_content" not in row_to_conversation(rows[-1])[-1]
        tok = _StubTokenizer()
        text = render_chat(tok, conv)
        assert "<think>\nthink 0\n</think>" in text and text.endswith("<|im_end|>\n")
        docs = tokenize_rows(tok, rows, log_every=0)
        assert len(docs) == 7
        total = sum(len(x) for x in docs)
        wins = pack(docs, 64, shuffle=True, seed=1)
        assert len(wins) == total // 65 and all(len(w) == 65 for w in wins)
        assert pack(docs, 64, shuffle=False) != wins or len(wins) == 0   # shuffle changes the stream (almost surely)

        # index rewrite for --install-dir
        index = {"metadata": {"total_size": 1}, "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
            "model.layers.0.mlp.up_proj.weight": "model-00001-of-00003.safetensors",
            "lm_head.weight": "model-00003-of-00003.safetensors",
            "mtp.fc.weight": "model-00003-of-00003.safetensors",
            "mtp.layers.0.self_attn.q_proj.weight": "model-00003-of-00003.safetensors"}}
        new, touched = rewrite_index(index, list(GGUF_NAME_HINT), HEAD_SHARD_NAME)
        assert touched == {"model-00003-of-00003.safetensors"}
        assert new["weight_map"]["mtp.fc.weight"] == HEAD_SHARD_NAME
        assert new["weight_map"]["lm_head.weight"] == "model-00003-of-00003.safetensors"
        assert all(hf_to_local(k) is None or v == HEAD_SHARD_NAME for k, v in new["weight_map"].items())

        # export command text mentions the exporter flag and the serve depth
        cmd = export_commands("/x/install", "/x/out")
        assert "--mtp --outtype q8_0" in cmd and "--spec-draft-n-max 3" in cmd

    try:
        import torch  # noqa: F401  (probe only)
        torch_note = "torch present (not exercised by --self-test)"
    except ImportError:
        torch_note = "torch absent - GPU path skipped, as designed"
    print("self-test OK: args, depth weights, tensor-name round trip, config plumbing, "
          f"depth alignment, JSONL/chat/packing, index rewrite, export command; {torch_note}")
    return 0


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("--model is required (or --self-test)")
    if args.install_dir and not args.data and not args.eval:
        _import_torch()
        if not args.head:
            ap.error("--install-dir without training needs --head")
        install_head(Path(args.model), Path(args.head), Path(args.install_dir))
        return 0
    if not args.data and not args.eval_data:
        ap.error("--data (or --eval-data with --eval) is required")
    if args.trunk_4bit and args.trunk_8bit:
        ap.error("--trunk-4bit and --trunk-8bit are mutually exclusive")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
