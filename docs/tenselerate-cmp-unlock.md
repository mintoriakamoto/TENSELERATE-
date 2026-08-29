# CMP 170HX memory unlock — what "80 GB stable" actually means

The CMP 170HX is a full GA100 die (the A100's silicon) with capacity and
compute fused/firmware-locked. The community `cmpunlocker` restores it in
software — no hardware mod. This is the operational note for the deployment
box, and the one honest caveat that matters.

## The unlock

- Tool: `cmpunlocker` (d3dx9 fork / amoghmunikote), technical wiki at
  Consensus-Protocol/cmp170hx.
- Target is set by `CMPUNLOCKER_TARGET`: `unlocked_80gb` (default),
  `unlocked_40gb` ("safer, fewer refresh issues"), `nativ_10gb` (revert).
- The unlock is **volatile** — lost on power cycle — and reapplied by a
  systemd daemon that rewrites the SS0/SS1 and CFG1/LMR registers via BAR0
  after any reset. So "stable" has two meanings: *persists across reboot*
  (the daemon handles this) and *the HBM actually works at that capacity over
  time* (the card decides this).
- PCIe: the card ships locked to Gen1; the unlock raises it to **Gen2**.
  Gen3/Gen4 are unsolved. This does not matter for us — decode keeps weights
  and KV resident, so nothing streams across PCIe per token.

## The caveat: 80 GB is not guaranteed

The two primary sources disagree, and the disagreement is the finding:

- The **tool** ships `unlocked_80gb` as the default and notes only that 40 GB
  has "fewer refresh issues."
- The **technical wiki** is blunter: for the 10 GB SKU (`10de:2082`) the
  80 GB configuration "was built, tested, and rejected as unstable," and that
  SKU's reliable target is 40 GB. The 8 GB SKU (`10de:20c2`) is the one that
  gives a solid 64 GB.

Making `nvidia-smi` **report** 80 GB is not proof that all 80 GB is stable
under sustained load. Refresh/ECC behaviour at the top of the range is where
10 GB cards fall over.

### Verify before relying on it

1. Confirm the report: `nvidia-smi --query-gpu=memory.total --format=csv,noheader`.
2. Soak the whole range, not a corner of it — allocate ~90% of the claimed
   VRAM and hammer it for hours (a memtest-style GPU stressor, or a real
   full-window serving load), watching for `Xid` errors, ECC volatile
   counts climbing, or throttling in `nvidia-smi -q`.
3. `tenselerate doctor` before every run — it fails loud if the driver isn't
   bound (on Blackwell the OPEN module is required; the CMP is Ampere so the
   standard module is fine).

## Why the requirement does not depend on 80 GB

This is the part that de-risks the whole thing. Weights are 15.41 GiB; KV is
bounded by the window, not the context, so 1M ctx is free at every capacity.
`tenselerate plan` at the 1M floor, per target:

| unlock target | widest window >= 400 tok/s | at the 32K quality floor |
| --- | --- | --- |
| 80 GB | 49K → ~453 tok/s | ~681 tok/s |
| 64 GB | 49K → ~424 tok/s | ~638 tok/s |
| **40 GB (documented-stable 10 GB SKU)** | 32K → ~502 tok/s | ~502 tok/s |

Every stable landing point clears **1M ctx and >400 tok/s**. If 80 GB proves
flaky, drop to `CMPUNLOCKER_TARGET=unlocked_40gb` and plan with
`--machine cmp170hx-40`: the requirement still holds, you just run at the 32K
window instead of having 49K headroom. Bandwidth is identical (same die);
only concurrency headroom changes.

**Bottom line:** unlock to 80 GB and verify it with a real soak — but the
product does not bet on it. `MACHINE_HW` carries `cmp170hx`, `cmp170hx-64`,
and `cmp170hx-40` so `plan` tells the truth for whichever one your card holds.

## Sources

- cmpunlocker (tool): https://github.com/d3dx9/cmpunlocker
- CMP 170HX technical wiki: https://github.com/Consensus-Protocol/cmp170hx
- Unlock walkthrough: https://bytwork.com/en/articles/cmp-170hx-unlock
- 80 GB reliability risk checklist: https://knightli.com/en/2026/07/22/cmp-170hx-80gb-memory-unlock-ai-gpu-buying-risk/
- Tom's Hardware coverage: https://www.tomshardware.com/pc-components/gpus/nvidia-crypto-mining-gpus-hacked-to-restore-locked-away-vram-in-order-to-feed-ai-boom-software-mod-unlocks-64gb-of-vram-on-usd250-cmp-170hx
