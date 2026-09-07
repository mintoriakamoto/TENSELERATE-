# CMP 170HX (40 GiB, full unlock) + RTX 3060 — measured results

Numbers measured on the `raven-9950x` box. Every figure here supersedes the
modeled constants in `tenselerate/cli.py` and the estimates in
`docs/rig-cmp170hx-3060.md`; if they disagree, this file wins.

| date | quantity | value | how | notes |
| --- | --- | --- | --- | --- |
| 2026-09-07 | 170HX FP16 tensor throughput | 162-170 TFLOPS | FP16 GEMM bench on the box | dense A100 class; the 256-cycle MMA gate does not reproduce on this unit |
| 2026-09-07 | 170HX power limit | 250 W | `nvidia-smi` | earlier 100 W cap lifted |
| 2026-09-07 | 170HX PCIe | Gen2 x4 | `nvidia-smi` | OTP-fused; ~2 GB/s |
| 2026-09-07 | decode, wrong build (capacity-only throttle workarounds) | ~22 tok/s | llama-server | baseline to beat; normal sm_80 build pending |

## Still to measure

- decode tok/s, normal build, 262K window, no spec (model prediction: ~45)
- decode tok/s with `--spec-draft-n-max 5` (model prediction: ~140-160)
- `svmi-bwprofile` achieved GB/s; `svmi-cmpbench` npl curve (should be flat)
- MTP acceptance per draft position (`svmi-bitspec`)
- UD-Q4_K_M vs mixed-INT8; q8_0 vs q4_0 KV at the 262K window
