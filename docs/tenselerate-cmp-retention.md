# Retaining the CMP 170HX 80 GB unlock — safely, at all costs

The honest frame first: **no software makes marginal HBM cells reliable.** 80 GB
stability is a property of your specific card's silicon. What is fully in our
control is the guarantee that *instability never reaches a served session* — so
"retain 80 GB at all costs" is implemented as **hold the card at 80 GB, prove
what is good, serve only that, and step capacity down before an answer ever goes
wrong.** The guardian is `tenselerate/engine/memguard.py`; this is the operator
procedure it enforces.

## Layer 1 — persistence (keep the unlock across resets)

The unlock is volatile; the cmpunlocker systemd daemon rewrites the capacity
registers (SS0/SS1, CFG1/LMR via BAR0) after every reset. Verify it is enabled
and survives a reboot:

```sh
systemctl status cmpunlocker         # must be enabled + active
sudo reboot; nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

If the daemon is healthy, capacity persistence is solved. This is necessary but
NOT sufficient — reporting 80 GB is not the same as 80 GB working.

## Layer 2 — ECC + hardware row-remapping (the real stability win)

GA100 is Ampere: it has **512 spare DRAM rows** and a hardware **row-remapper**
that retires degrading cells and — once applied — **persists for the life of the
card across resets**. That mechanism is the single biggest lever for holding
80 GB, but it only fires when **ECC is enabled** to detect the errors that
trigger a remap. Try to enable it:

```sh
sudo nvidia-smi -e 1                  # enable ECC; needs a GPU reset/reboot
sudo nvidia-smi -q -d ECC,ROW_REMAPPER
```

- **If ECC turns on:** you have live bad-cell retirement. Watch
  `Correctable`/`Uncorrectable` volatile counts and `Remapped Rows`. The
  guardian treats an exhausted remap pool (512 used) or any uncorrectable error
  as disqualifying.
- **If ECC will not enable** (it is on the unlock's unresolved list, so this is
  likely on some cards): you have no live retirement, and Layer 3's soak
  carve-out becomes the whole safety net. The guardian marks the card
  `ECC OFF - relying on soak carve-out` so nobody forgets.

## Layer 3 — soak, then carve out the unstable tail

A capacity screenshot is not proof. Exercise the WHOLE claimed range under load
long enough for refresh drift to show (`SoakPlan`: ~95% of VRAM, ~6 h, zero
fatal errors, zero SBE at the top of range):

```sh
# fill ~90% of the claimed VRAM and hammer it for hours, e.g. a GPU memtest
# or a real full-window serving load, while watching for trouble:
watch -n5 'nvidia-smi -q -d ECC,ROW_REMAPPER,PAGE_RETIREMENT | \
           grep -Ei "Xid|Uncorrectable|Remapped|Pending"'
dmesg -w | grep -i xid                # any Xid during the soak = a bad region
```

Whatever fails is the unstable tail. **Carve it out**: per the community
procedure, test the memory banks and disable the faulty ones per card, or simply
cap usable capacity below the failing region. Feed the guardian:
`MemoryHealth(reported_gib=80, verified_gib=<soaked>, quarantined_gib=<bad>)`,
and it budgets `usable = verified - quarantined` and picks the largest stable
tier.

## The payoff — retention never costs the product

The guardian degrades capacity, not correctness. And because the requirement is
met at every stable tier, a card that cannot hold a full clean 80 GB still
serves within spec:

| verified-usable | tier the guardian serves | 1M ctx + >400 tok/s |
| --- | --- | --- |
| ≥ 80 GB | `cmp170hx` (49K win → ~453 tok/s) | met |
| 64–79 GB | `cmp170hx-64` (49K → ~424 tok/s) | met |
| 40–63 GB | `cmp170hx-40` (32K → ~502 tok/s) | met |
| < 40 GB | **refuses to serve** (`MemoryUnsafe`) | — |

So "at all costs" is honoured the only way that is real: the card stays unlocked
to 80 GB, the engine serves every byte it can prove good, and it steps down to a
verified tier — or refuses — rather than serve a single token from memory that
might be lying. `choose_machine_profile(health)` turns a card's health straight
into the `plan`/serve profile, so the whole pipeline budgets proven memory.

## Runtime guard (roadmap)

`doctor` runs Layers 1–2 as a pre-flight; the soak is an operator step. Next:
a live guardian thread that polls `nvidia-smi -q -d ECC` during serving and, on
a new uncorrectable error or Xid, quarantines the region and re-plans at the
next-lower tier without dropping in-flight sessions (their bounded footprints
migrate — see the mesh). That is retention as a running property, not just a
bring-up check.

## Sources

- NVIDIA GPU Memory Error Management (row-remapping, page retirement, ECC): https://docs.nvidia.com/deploy/pdf/NVIDIA-GPU-Memory-Error-Management.pdf
- A100 row remapping: https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html
- CMP 170HX unlock + per-card bad-bank procedure: https://bytwork.com/en/articles/cmp-170hx-unlock
- 80 GB reliability risk checklist: https://knightli.com/en/2026/07/22/cmp-170hx-80gb-memory-unlock-ai-gpu-buying-risk/
