# What to take from ExLlama, vLLM, SGLang and TensorRT-LLM

Ranked against *this box's measured bottlenecks*, not against a feature list.
Every other engine has good ideas; most of them solve problems we do not have.
The decode step here is:

| term | measured | physics | gap |
| --- | --- | --- | --- |
| weight read, 15.4 GB | 18.5 ms | 10.3 ms @ 1493 GB/s | 1.8x |
| everything else at batch 1 | **11.4 ms** | ~0.3 ms | **38x** |
| MMQ floor, widths 2-16 | ~55 ms | ~17 ms | 3.2x |
| per live sequence | 5.6 ms | ~0.9 ms | 6x |
| KV read at 262K | 51 ms | 9.5 ms | 5.4x |

So: launch overhead, small-width tiles, per-sequence cost. Borrow against those.

---

## Tier 1 - the 11.4 ms residual at batch 1

### 1. Graph-stable decode inputs (vLLM, TensorRT-LLM) — the highest-value borrow

llama.cpp captures CUDA graphs, but `ggml_cuda_graph_update_required`
(`ggml-cuda.cu`) rebuilds when *anything* moves: for every node it memcmp's the
whole `ggml_tensor` plus each source's `data` pointer, `ne` and `nb`. One
changed pointer anywhere in a 64-layer graph marks the whole thing dirty and
forces a `cudaGraphExecUpdate`, or a full re-instantiate when that update fails.

A hybrid GDN decode step is exactly the shape that trips this: the recurrent
state cell for a sequence rotates, the KV cache head advances, and views move
with them. If those pointers change every step, the graph is rebuilt every step
and the launch overhead the graph exists to remove comes back.

**vLLM's answer, which is the borrowable part:** decode runs on *persistent
fixed-size buffers*. Slot mappings, block tables and positions live in tensors
whose addresses never change; only their **contents** are rewritten in place
each step. The graph then captures once per batch size and is replayed with no
update at all.

Applied here that means: the GDN state and KV access should read a constant
base pointer plus an index tensor updated in place, never a re-pointed view.
Upstream already moves in this direction (`s_copy`,
`state_snapshot_src_idxs`/`_dst_idxs` are index tensors); the question is
whether any pointer still moves per step.

**Cost to find out: 30 seconds.** `llama-bench -n 64` with and without
`GGML_CUDA_DISABLE_GRAPHS=1`. Identical numbers mean graphs are already
inert for this model and up to 11 ms per token is sitting on the table. That
is issue #53, and it should be the first thing run on the box.

### 2. Kernel fusion in the GDN block (ExLlamaV2/V3)

ExLlama's speed comes substantially from fusing what other engines leave as
separate launches: QKV as one matmul, MLP gate/up fused, norms folded into the
following GEMM. The GDN block here is ~15 kernels per layer across 48 layers.
Even partial fusion (conv + gate + norm) cuts launches proportionally, and
launches are what the 11.4 ms is made of. Profile first (#53) so fusion targets
the launches that actually cost.

---

## Tier 2 - the 55 ms MMQ floor at widths 2-16

### 3. Shape-specialised tiles / autotuning (MLC-TVM, CUTLASS, TensorRT-LLM)

Upstream's MMQ tile configuration is tuned per architecture but not per *shape*,
and Ampere inherits Turing's row. A 55 ms floor that does not move between
width 2 and width 16 is the signature of a tile shape chosen for large N. The
borrowable technique is what MLC and TensorRT-LLM do: enumerate a small tile
space per (type, width) and pick by measurement rather than by constant.

This is also where the Ada-gate back-port already landed (#71): width 2 is an
MTP verify pass, and Ampere currently drops it to the MMA kernel.

---

## Tier 3 - the only lever that moves the 95 tok/s ceiling

### 4. Trellis / codebook quantization (ExLlamaV3 EXL3, QTIP)

At batch 1 the card must read every weight once per token. 15.4 GB at 1493 GB/s
is 95 tok/s and no scheduling changes that. The only way through it is fewer
bytes per weight.

EXL3 reaches roughly Q4_K_M quality near 3.5 bpw using a trellis codebook with
a Hadamard-style incoherence transform, against Q4_K_M's ~4.8 bpw. That is
about **27% fewer bytes per token**, so about 27% faster at batch 1, for a
quality cost that has to be measured rather than assumed. It is real work — a
new quant type, a CUDA dequant path, and a conversion — and it is the only item
on this page that raises the physics ceiling instead of closing the gap to it.

---

## Tier 4 - multi-agent serving, not raw tok/s

### 5. RadixAttention (SGLang)

A radix tree of KV prefixes shared automatically across *all* requests. This
repo now has three narrower versions of the same idea: LCP slot selection, the
host-RAM prompt cache, and the on-card `fork_from` (#67/#70). RadixAttention is
their superset: any request automatically reuses any prefix any other request
already has, with no slot id passed by the client. Worth taking once the fork
primitive has box numbers.

SGLang also caches **recurrent states at prefix boundaries** for hybrid models,
which is precisely the prefix-checkpoint half of #67 still unbuilt.

### 6. Chunked prefill (vLLM)

Prefill chunks are interleaved with decode in the same batch, so a long prompt
does not stall every other stream. With 8-16 agents and a 35K system prompt,
one agent's cold start currently blocks the rest. llama.cpp has micro-batching
(`-b`/`-ub`) but not vLLM's scheduler-level interleaving.

---

## Deliberately not borrowed

| idea | why not here |
| --- | --- |
| PagedAttention (vLLM) | solves VRAM fragmentation; our KV is slot-based and the bounded window (#62) makes per-slot capacity fixed. Pages would add indirection to fix a problem we removed |
| Tree / Medusa speculation | trades breadth for acceptance. Ours is already 0.88 at depth 1 greedy; depth is the constraint, not branch coverage |
| FlashAttention-3 | Hopper wgmma + TMA. sm_80 cannot run it (`docs/ampere-backports.md`) |
| Continuous batching | already here (`-cb`, default) |
| Prefix caching by hash | superseded for our workload by #5 above |

---

## Order

1. **#53, the graph A/B.** 30 seconds, and it decides whether item 1 is worth
   up to 11 ms per token or nothing.
2. **#54 + profile.** Fusion and tile work, aimed by the profile rather than guessed.
3. **#56.** The MTP head, which multiplies whatever 1 and 2 leave.
4. **EXL3-style quantization.** The ceiling lever; large, and worth scoping only
   once the gap to the current ceiling is closed.

Nothing here is adopted on reputation. Each becomes a row in
`benches/cmp170hx-3060/README.md` next to the number it was supposed to move.
