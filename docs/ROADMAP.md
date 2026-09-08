# Roadmap

What is being worked on, in order, and how each item is judged done. Every
item is a GitHub issue; this page is the ordering and the reasoning.

Target: the CMP 170HX (40 GiB) + RTX 3060 box serving a 27B Q4_K_M model to
Hercules, with the most parallel agents at the longest context the card can
hold. The method for that is `docs/bounded-window-serving.md`. Numbers to beat: **33.5 tok/s** single stream, **70.5 tok/s**
aggregate at 4 x 256K, **12.4 tok/s** at 262K depth. Everything below is
measured against those on the release binary, not a dev build.

## Now (this month)

| # | Item | Why first | Done when |
| --- | --- | --- | --- |
| [#67](https://github.com/mintoriakamoto/TENSELERATE-/issues/67) | Fork, don't fetch: on-card sequence fork for delegation and a prefix checkpoint | Every other way to start an agent moves bytes over the 2 GB/s link; a fork moves none and gives the child the parent's whole context | Delegation child starts in one decode step; 16 pinned-prompt slots fit; `docs/bounded-window-serving.md` |
| [#63](https://github.com/mintoriakamoto/TENSELERATE-/issues/63) | Bounded attention window (PR #62): grade the slot table and the needle test | The one lever that turns the depth penalty into slots: 16 x 32K or 9 x 64K unbounded sequences instead of 2 x 256K capped | Rows in the benches README next to the predictions in `docs/bounded-window-serving.md`; default decided |
| [#60](https://github.com/mintoriakamoto/TENSELERATE-/issues/60) | Run the open-items bench experiments | Fills every "predicted" row with a number; every other item depends on these numbers | "Still to measure" in the benches README has no empty rows |
| [#53](https://github.com/mintoriakamoto/TENSELERATE-/issues/53) | Profile the GDN block at batch 1 (`nsys`) | The ~11 ms non-weight residual is launch count by the arithmetic; measure before writing a kernel | Launch count and ms-per-launch recorded in `docs/kernel-work.md` §3 |
| [#57](https://github.com/mintoriakamoto/TENSELERATE-/issues/57) | Grade GQA-packed attention at 262K | Predicted 12.4 to ~20 tok/s; the code is in, only the A/B is missing | Gate default set from measured rows |
| [#59](https://github.com/mintoriakamoto/TENSELERATE-/issues/59) | Weekly upstream sync | Keeps the next merge small; the first one cost 73 conflicts | Tracking issue; each sync PR links it |

## Next

| # | Item | Why | Done when |
| --- | --- | --- | --- |
| [#54](https://github.com/mintoriakamoto/TENSELERATE-/issues/54) | MMQ small-width floor and the MMVQ crossover | Multi-slot aggregate is capped by the 55 ms MMQ floor at widths 2-16 | `MMVQ_MAX` default chosen from data; benches row |
| [#55](https://github.com/mintoriakamoto/TENSELERATE-/issues/55) | N=32 slot sweep | Predicted 170-220 aggregate; grades the width model at the top end | N=32 row next to the prediction |
| [#56](https://github.com/mintoriakamoto/TENSELERATE-/issues/56) | MTP head that survives sampling | Greedy gives 46.2 tok/s at 0.88 acceptance but loops; sampled acceptance is 0.22 | Acceptance > 0.5 at the serving temperature, tok/s above 33.5 |
| [#58](https://github.com/mintoriakamoto/TENSELERATE-/issues/58) | RTX 3060 side model for delegation | Splitting the 27B across cards cannot win (PCIe); a second model can | `HERCULES.md` side-model section with numbers |

## Not planned

- Splitting the 27B across the 170HX and the 3060: PCIe-bound, predicted
  -25%, and it OOMs. See the benches README.
- Tensor-core paths on the 170HX: FP16 is fused off; only dp4a / MMQ exist.
- Re-adding the fork's own dspark drafter or the recurrent snapshot ring:
  upstream's `draft-dspark` and rollback replaced both (#52).

## How items move

Branch from `main`, PR with the template's verification checklist, CI green,
merge. The `main` push cuts the release; the box installs it with
`FLAVOR=cuda scripts/tenselerate-update.sh --binary` and the number goes into
`benches/cmp170hx-3060/README.md` next to the prediction it grades.
`docs/dev-workflow.md` has the loop in full.
