# Long context without RoPE scaling — the methods, and where this engine stands

The no-RoPE-scaling rule is doctrine here: no YaRN, no position interpolation,
no NTK-aware stretching, ever (`needs_rope_scaling()` enforces it). This maps
the known long-context methods against that rule: what is banned by it, what
this engine already runs, what it adopts now, and what remains a candidate.

## Banned: the positional-remapping family

All of these buy context by making the model attend at positions it was never
trained on — exactly the quality trade the rule exists to refuse:

| method | mechanism | why banned |
| --- | --- | --- |
| Position Interpolation | compress positions linearly | extrapolation by another name |
| NTK-aware / dynamic NTK | stretch RoPE base frequency | same, frequency domain |
| YaRN | uneven per-band interpolation | best of the family; still a remap |
| LongRoPE / LongRoPE2 | searched per-dim rescale + fine-tune | needs retraining to hide the cost |
| Self-Extend / Dual-Chunk | grouped/re-used positions at inference | positional aliasing, quality loss on recall |

## Already running: the architectural answer

The engine's core design *is* the strongest known no-remap method — the hybrid
that Qwen3-Next introduced and `qwen3_5` ships:

- **48 Gated-DeltaNet linear layers**: fixed-size recurrent state, **no
  positional encoding at all** — context is unbounded by construction. This is
  the NoPE idea taken to its conclusion: the long range lives in a state, not
  in positions.
- **16 full-attention layers on a bounded window** (>= the 32K quality floor,
  <= the 262K trained range): every attended position is one the model was
  trained on. KV is constant (~4.25 GiB at the 128K default), which is what
  makes the 1M context floor and the 600 tok/s speed floor coexist.
- **Paged KV + continuous batching** (`engine/kvpool.py`, `engine/scheduler.py`):
  the serving-side half of the same story.

## Adopted now: attention sinks

StreamingLLM's finding (arXiv:2309.17453): softmax attention parks surplus
probability mass on the first few tokens; a sliding window that evicts them
collapses in quality, and pinning ~4 of them recovers it out to millions of
tokens. That is a pure quality-retention method with no positional remap —
sink cache positions are re-anchored to 0..3, so the attended span is
window + sinks and never leaves the trained range.

In this engine: `ATTENTION_SINK_TOKENS = 4` in `config.py`,
`ModelConfig.attention_sink_tokens` counted into `resident_kv_tokens` and into
the no-extrapolation check. Cost: ~136 KiB of KV per sequence; at the 32K
window it rounds one block up and costs one concurrency slot (44 -> 43,
~634 tok/s — still above the speed floor).
Pinned by `tests/tenselerate/test_attention_sinks.py`.

## Candidates, in order of expected value

1. **MTP self-speculation** (roadmap phase 3): the model's own draft head,
   2-3 tokens per verify step. Multiplies throughput ~1.5-2x with *identical*
   outputs — the only speed lever with provably zero quality cost.
2. **KV cache precision A/B** (q8_0 vs q4_0/FP8): q4_0 halves KV per sequence
   (double concurrency), but it is a quality trade and therefore gated on a
   measured A/B, not adopted by default. FP8 is the same footprint as q8_0, so
   its case would be accuracy, not capacity.
3. **Heavy-hitter / landmark retention** (H2O, SnapKV, Ltri-LLM style): keep
   the highest-attention tokens beyond the window instead of a strict slide.
   Same spirit as sinks — spend a little resident KV where the mass actually
   is. Worth prototyping once the attention kernel lands; the win over
   sinks + window is unproven for this hybrid, where the GDN state already
   carries the long range.
4. **Compressive memory (Infini-attention style)**: fold evicted KV into a
   summary state. Largely redundant here — the 48 GDN layers *are* that
   memory — but the idea may apply inside the 16 full layers.

Not applicable: Ring/blockwise attention (scales compute across devices, not
positions — orthogonal to the rule), Mamba conversion (we already run a
trained hybrid; converting the 16 full layers would be a different model, and
the model is locked).

## Sources

- StreamingLLM / attention sinks: https://arxiv.org/abs/2309.17453
- Context-extension survey: https://arxiv.org/abs/2402.02244
- Qwen3-Next hybrid (GDN + gated attention, 3:1): https://vllm.ai/blog/2025-09-11-qwen3-next
- Gated DeltaNet: https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/
- Sliding-window adaptation trade-offs: https://arxiv.org/abs/2512.10411
