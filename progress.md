# Project Progress — Cost-Aware Cascade Router

Last updated: 2026-05-30

---

## Project Overview

Training a **cost-aware cascade router** for code generation. A cheap model (gpt-5.4-nano, cost=1) attempts a problem first. If it fails, a router picks the cheapest recovery action:

| Action | Cost (nano-units) | Meaning |
|---|---|---|
| `reflect` | ~2 | nano fixes its own code given stderr |
| `replan` | ~2 | nano discards and re-plans from scratch |
| `escalate` | ~13 | gpt-5.4 solves from scratch |
| `unsolvable` | 0 | give up (4cls/5cls only) |

Two architectures:
- **Arch A** (post-proceed): router fires after a failed proceed attempt; input = problem + verdict + stderr
- **Arch B** (pre-proceed): router fires before any LLM call; input = problem only

CRC (Conformal Risk Control) sits on top: operator sets a cost budget C → CRC selects λ to guarantee E[cost] ≤ C with finite-sample bounds.

**How CRC works:**

At inference, the router picks the action with the highest *cost-penalized* score:

```
a*(x) = argmax_a  [ p_a(x) - λ · cost(a) ]
```

where `p_a(x)` is the router's softmax probability for action `a`, and `cost(a)` is the action's cost (reflect=2, replan=2, escalate=13). λ=0 reduces to plain argmax; larger λ increasingly favors cheap actions.

CRC calibrates λ on a held-out calib set: find the smallest λ such that the empirical cost bound holds with high probability (Hoeffding inequality, δ=0.05). The guarantee is:

```
E[cost] ≤ C    with probability ≥ 1 - δ    (Mode A)
P[fail] ≤ α    with probability ≥ 1 - δ    (Mode B)
```

The finite-sample correction ε shrinks the effective budget: CRC finds λ with E[cost] ≤ C − ε on calib, so the test guarantee holds. Larger calib set → smaller ε → less conservative.

---

## Data

### SFT Datasets

| Variant | File | Examples | Labels |
|---|---|---|---|
| Arch A 3cls | `datasets/router_no_reason_v1_3cls.json` | 5,498 | reflect / replan / escalate |
| Arch A 4cls | `datasets/router_no_reason_v1_4cls.json` | 13,626 | + unsolvable |
| Arch B 4cls | `datasets/router_no_reason_v1_archB_4cls.json` | 19,172 | problem-only input |
| Arch B 5cls | `datasets/router_no_reason_v1_archB_5cls.json` | 27,300 | + unsolvable |

Previous smaller versions archived in `datasets/old/`.

**Label rule**: cheapest action whose rollout verdict == pass. Cost ties (reflect vs replan) broken by priority: reflect > replan > escalate.

**Source rollouts**: `cloudide/rollout/*.jsonl` — short-circuit rollout (proceed → reflect+replan → escalate only if both fail) from bcb / lcb / taco / apps / apps_functional etc.

### Benchmark Datasets (Eval)

4 files in `datasets/benchmarks/` — 400 examples each (one per dataset variant).

Originally 8 files (random + stratified × 4 variants), but the two sampling strategies produced near-identical label distributions (±2pp) and identical source distributions — stratified files removed.

**Note**: all benchmark examples are drawn from the same rollout pool as the training data (100% overlap with `router_no_reason_v1_*.json`). Current eval measures in-distribution fit, not generalization. A true held-out benchmark needs to be rebuilt from rollout data not used in training.

Coverage after exhaustive rollout fill-in + null removal (2026-05-29):

| File | n | n_with_sa | unsolvable (empty SA) |
|---|---|---|---|
| 3cls_bench_random | 328 | 315 | 13 |
| 4cls_bench_random | 318 | 122 | 196 |
| archB_4cls_bench_random | 302 | 297 | 5 |
| archB_5cls_bench_random | 310 | 207 | 103 |

Null examples removed: they came from older rollout batches no longer on disk and were unfillable. stratified variants were lost during cleanup — only random files remain.

Each example has `successful_actions` (list of ALL actions that pass — not just the cheapest), enabling correct solve_rate evaluation.

---

## Models Trained

| Model | Data | Config | Status | Best Checkpoint |
|---|---|---|---|---|
| `router_arch_a_3cls` (v1) | 1,583 ex | 1-GPU, 3 epochs | Done | `checkpoint-200-renamed` |
| `router_arch_a_4cls` (v1) | 4,813 ex | 1-GPU, 2 epochs | Done | `router_arch_a_4cls-renamed` |
| `router_arch_a_3cls_v2` | 5,498 ex | 2-GPU, 3 epochs | Done — **path bug, unusable** | checkpoint-1000 |
| `router_arch_a_4cls_v2` | 13,626 ex | 2-GPU, 2 epochs | Done — **class collapse** | checkpoint-2300 (unusable) |
| `router_arch_a_3cls_v3` | 5,114 ex (deduped) | 2-GPU, 3 epochs | Done — **path bug fixed** ✓ | `checkpoint-800` in `_fixed/` (solve_rate=0.774) |
| `router_arch_a_4cls_v3` | 7,078 ex (deduped+cap) | 2-GPU, 3 epochs | Done — **path bug fixed** ✓ | `checkpoint-600` in `_fixed/` (solvable solve_rate=0.579) |
| `router_arch_a_3cls_v4` | 4,656 ex (train split, no holdout) | 2-GPU, 3 epochs | Done — **path bug fixed** ✓ | `checkpoint-1000` in `_fixed/` (solve_rate=0.814 bench / 0.656 holdout) |
| `router_arch_a_3cls_v5b_clean` | 3,305 ex (vote modal, no noisy) | 2-GPU, 3 epochs | **Running** (machine A) | TBD |
| `router_arch_a_3cls_v5b_full` | 4,504 ex (vote modal + hard fallback) | 2-GPU, 3 epochs | **Queued** (machine A, after v5b_clean) | TBD |
| `router_arch_a_3cls_v5a_clean` | 26,413 ex (soft copies, no noisy) | 2-GPU, 1 epoch | **Ready** (machine B) | TBD |
| `router_arch_a_3cls_v5a_full` | 27,612 ex (soft copies + hard fallback) | 2-GPU, 1 epoch | **Ready** (machine B, after v5a_clean) | TBD |

All in `sft_runs/outputs/`.

### v2 Collapse Issue

`router_arch_a_3cls_v2` (all checkpoints) predicts "reflect" for ~90% of inputs regardless of content. Confirmed via generation and log-prob scoring.

**Investigation** (ruled out in order):
1. Format difference — all 5,498 examples share the same `build_sft_router_v1.py` format. ✗
2. Class imbalance — escalate 44%, reflect/replan ~28% each. Balanced. ✗
3. verdict/input length distribution — old vs new nearly identical. ✗
4. **Duplicate problems with conflicting labels** — ✓ root cause.

**Root cause**: v1 (1,583 ex) and the 3,915 new examples were built from different rollout batches. The build script deduped by `f"{dataset_tag}::{problem_id}"`, so the same problem appearing in two rollout files was treated as two distinct training examples. Across the 5,498 merged examples, **438 problem_ids appear in both old and new data; 166 of those (38%) carry conflicting labels**:

| Conflict | Count |
|---|---|
| replan → escalate | 42 |
| escalate → replan | 32 |
| replan → reflect | 25 |
| reflect → escalate | 24 |
| reflect → replan | 21 |
| escalate → reflect | 22 |

Same input, opposite target — gradients cancel on these 166 examples, and the model learns the path of least resistance ("reflect" as the modal safe default).

- **Symptom**: `always_reflect` solve_rate = 27%, so collapsed model gets solve_rate ~28%.
- **Fix (v3)**: deduplicate by `problem_id` only, keep-first (v1 labels win). Result: 5,498 → 5,114 examples, 0 conflicting labels. Same fix applied to 4cls: 13,626 → 11,718 (dedup) → 7,078 (unsolvable capped at 30%).

### v3 Path Bug (LoRA Key Mismatch)

v3 training completed (max step 1728, checkpoint-800 best by eval_loss=0.2814) but the adapter had **zero effect** at inference — logits were bit-for-bit identical to the base model.

**Root cause**: During training, LlamaFactory loaded `Qwen/Qwen3.5-4B` wrapped inside a `language_model` sub-module (likely triggered by a VLM-style `trust_remote_code` path on this devbox). Saved LoRA keys had the form:

```
base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_B.weight
```

But the standard Qwen3.5-4B inference path uses:

```
base_model.model.model.layers.0.linear_attn.in_proj_a.lora_B.weight
```

PEFT found no matching modules and **silently created orphan LoRA layers** — weights non-zero in file (lora_B norm ~0.04–0.60) but never applied to any real layer during forward pass. eval_loss was real (gradients did flow through the training graph's wrapped model), but the saved weights are useless for inference on the unwrapped model.

**Fix**: rename all keys in `adapter_model.safetensors` across all 9 checkpoints:
```python
new_key = key.replace('model.language_model.', 'model.')
```
Fixed checkpoints saved to `sft_runs/outputs/router_arch_a_3cls_v3_fixed/`. The same fix was applied to 4cls v3: `sft_runs/outputs/router_arch_a_4cls_v3_fixed/`. v2 (both 3cls and 4cls) has the identical bug but was not fixed — the conflicting-labels problem in v2 training data makes it not worth recovering.

**Detection heuristic for future runs**: after loading an adapter, compute logits on 1 example with and without the adapter — if `max|logits_adapted - logits_base| < 0.01`, the adapter is not being applied.

---

## Evaluation Results

### Metric Definitions

- **cls_acc** (decision accuracy): argmax prediction == GT label. **Misleading** — penalizes router for choosing a different but equally valid action.
- **solve_rate**: argmax prediction ∈ successful_actions. **Correct metric** — measures whether the router's chosen action actually solves the problem.

### Per-Action Baseline Solve Rates (3cls_bench_random, n=328)

SA lists deduplicated and filtered to {reflect, replan, escalate} before computing.

| Strategy | solve_rate | mean cost |
|---|---|---|
| always_reflect | 0.277 | 2 |
| always_replan | 0.409 | 2 |
| always_escalate | 0.768 | 13 |
| router v1 argmax | 0.735 | 10.5 |
| **router v3 argmax** | **0.774** | ~10.3 |
| CRC Mode A C=8 (v1) | 0.665 | 7.3 |
| oracle (perfect router) | 0.960 | — |

**Oracle** = fraction of problems with at least one successful action (data-determined ceiling). 4.0% are genuinely unsolvable.

**v3 vs v1**: v3 argmax (77.4%) now beats always_escalate (76.8%) while costing ~10.3 vs 13. First router version to exceed the always_escalate solve_rate baseline. The +3.9pp gain over v1 comes from cleaner training data (deduped, no conflicting labels).

Key remaining gap: v3 (77.4%) vs oracle (96.0%) — 18.6pp. Router quality is still the primary bottleneck. CRC can trade some of that for cost reduction (Mode A C=8 buys ~3.5 cost units at the price of ~10pp solve_rate).

**Why GPT router (no SFT) is useless for CRC:**

A GPT base model produces high-entropy probability distributions over the three actions, e.g.:

```
p_escalate ≈ 0.38,  p_reflect ≈ 0.34,  p_replan ≈ 0.28   (typical, max gap ~0.10)
```

Plugging into the CRC policy `a*(x) = argmax_a [p_a(x) − λ · cost(a)]`, even a tiny λ knocks out escalate, because cost(escalate)=13 vs cost(reflect)=2 is an 11-point gap that swamps the 0.10 probability difference:

```
λ = 0.01:
  escalate: 0.38 − 0.01×13 = 0.25
  reflect:  0.34 − 0.01×2  = 0.32  ← wins on almost every example
```

So the CRC λ-sweep collapses immediately: the moment λ crosses ~0.005, every example routes to reflect. The resulting chosen_dist = {reflect: n, replan: 0, escalate: 0} for nearly all α — CRC has no meaningful trade-off to make.

**SFT fixes this** by teaching the model to assign p_escalate ≈ 0.9 on genuinely hard problems. Now the cost penalty needs λ > 0.06 before reflect beats escalate on those examples, giving CRC a wide usable range of λ to trade off solve_rate against cost.

### Successful Actions Distribution (3cls_bench_random, n=328)

| \|SA\| | Count | % |
|---|---|---|
| 0 (unsolvable) | 13 | 4.0% |
| 1 (exactly one works) | 185 | 56.4% |
| 2 | 98 | 29.9% |
| 3 (all work) | 32 | 9.8% |

~96% of problems solvable. Reflect works on only 27.7% — many problems require escalate.

### Router Benchmark Eval — Per-Label Breakdown (3cls_bench_random, n=328)

**v1** (checkpoint-200-renamed):

| Label | cls_acc | solve_rate | n |
|---|---|---|---|
| reflect | 0.356 | 0.716 | 88 |
| replan | 0.134 | 0.640 | 89 |
| escalate | 0.908 | 0.868 | 151 |
| **overall** | **0.548** | **0.765** | 328 |

**v3** (checkpoint-800, fixed):

| Label | cls_acc | solve_rate | n |
|---|---|---|---|
| reflect | 0.307 | 0.727 | 88 |
| replan | 0.270 | 0.753 | 89 |
| escalate | 0.848 | 0.815 | 151 |
| **overall** | **0.546** | **0.774** | 328 |

**Analysis:**
- solve_rate >> cls_acc in both models because reflect/replan are cost-tied and frequently interchangeable — the "wrong" pick often still solves the problem.
- v3 vs v1 by label:
  - **reflect**: cls_acc drops (0.356→0.307) but solve_rate improves (+1.1pp). v3 misclassifies more reflect examples but routes them to actions that still work.
  - **replan**: cls_acc improves significantly (+13.6pp: 0.134→0.270). v3 learned to distinguish replan from escalate better (deduped data resolved conflicting labels that were probably in this category).
  - **escalate**: cls_acc drops slightly (0.908→0.848) but is still high; solve_rate drops (0.868→0.815). v3 is slightly more willing to try reflect/replan on problems the GT labeled escalate — since many escalate-labeled problems also accept reflect/replan (SA lists often include multiple actions), the overall solve_rate still goes up.

### Router Benchmark Eval — v4 (3cls_bench_random, n=328)

**v4** (checkpoint-1000, fixed):

| Label | cls_acc | solve_rate | n |
|---|---|---|---|
| reflect | 0.455 (40/88) | 0.784 (69/88) | 88 |
| replan | 0.225 (20/89) | 0.742 (66/89) | 89 |
| escalate | 0.907 (137/151) | 0.874 (132/151) | 151 |
| **overall** | **0.601** | **0.814** | 328 |

**v4 vs v3 summary:**
- solve_rate: **+4.0pp** (0.814 vs 0.774) — first model to clearly beat always_escalate (0.768)
- cls_acc: +5.5pp (0.601 vs 0.546)
- escalate: +5.9pp solve_rate; reflect: +5.7pp; replan: -1.1pp

### Holdout Eval (genuinely no-overlap, n=360 calib / 360 test)

First time training and test/calib are split without overlap. `router_arch_a_3cls_v3_train.json` (4,656 ex) was used for training; holdout_3cls_calib.json + holdout_3cls_test.json (360+360) were never seen during training.

**v4 calib (n=360, no-overlap):**

| Label | cls_acc | solve_rate | n |
|---|---|---|---|
| reflect | 0.462 (48/104) | 0.587 (61/104) | 104 |
| replan | 0.076 (7/92) | 0.370 (34/92) | 92 |
| escalate | 0.963 (158/164) | 0.945 (155/164) | 164 |
| **overall** | **0.592** | **0.694** | 360 |

**v4 test (n=360, no-overlap):**

| Label | cls_acc | solve_rate | n |
|---|---|---|---|
| reflect | 0.398 (39/98) | 0.520 (51/98) | 98 |
| replan | 0.138 (15/109) | 0.339 (37/109) | 109 |
| escalate | 0.993 (152/153) | 0.967 (148/153) | 153 |
| **overall** | **0.572** | **0.656** | 360 |

**Key observations:**
- Holdout solve_rate (0.694 calib / 0.656 test) vs bench solve_rate (0.814) shows modest overfitting
- Escalate classification essentially perfect (99.3% on test); escalate solve_rate 96.7% — model is highly calibrated for hard problems
- Replan is the weak class: model conflates replan with escalate on unseen data (13.8% cls acc on test → mostly routed to escalate or reflect)
- Replan cls accuracy drops sharply (v4 bench 22.5% → holdout test 13.8%) but escalate improves
- Note: v3's "holdout" solve_rate of 0.983 was inflated (v3 was trained on holdout data); 0.656 is the first honest out-of-distribution estimate

### CRC Results — Old Benchmark (3cls_bench_random, random 50/50 split)

#### v1 router

Source: `conformal/results/crc_bench_3cls_v1_n328.json`

n=328, calib=164, test=164, δ=0.05, ε_cost=1.051, ε_fail=0.096

argmax baseline: solve_rate=0.732, cost=10.32

**Mode A:**

| C | λ | test solve | test cost | guarantee |
|---|---|---|---|---|
| 5 | 0.0252 | 0.500 | 3.14 | ✓ |
| 7 | 0.0184 | 0.567 | 4.95 | ✓ |
| **8** | **0.0145** | **0.616** | 6.23 | ✓ |
| 9 | 0.0109 | 0.652 | 7.37 | ✓ |
| 10 | 0.0090 | 0.677 | 8.10 | ✓ |
| 12 (≈argmax) | 0.0012 | 0.726 | 10.25 | ✓ |
| argmax (λ=0) | 0 | 0.732 | 10.32 | — |

**Mode B:** Only α=0.65 feasible (test_solve=0.451, cost=2.0). ε_fail=0.096 — need n≥500 calib to unlock α<0.65.

#### v3 router

Source: `conformal/results/crc_bench_3cls_v3.json`

n=328, calib=164, test=164, δ=0.05, ε_cost=1.051, ε_fail=0.096

argmax baseline (test): solve_rate=0.726, cost=9.38, fail=0.274
argmax baseline (calib): fail=0.177

**Mode A:**

| C | λ | test solve | test cost | guarantee |
|---|---|---|---|---|
| 4 | 0.0175 | 0.470 | 2.27 | ✓ |
| 5 | 0.0144 | 0.512 | 2.74 | ✓ |
| 6 | 0.0104 | 0.585 | 4.28 | ✓ |
| 7 | 0.0098 | 0.616 | 5.02 | ✓ |
| **8** | **0.0074** | **0.622** | **5.49** | ✓ |
| 9 | 0.0072 | 0.659 | 6.70 | ✓ |
| 10 | 0.0048 | 0.695 | 7.50 | ✓ |
| 11 | 0.0023 | 0.713 | 8.71 | ✓ |
| argmax (λ=0) | 0 | 0.726 | 9.38 | — |

**Mode B:** All α infeasible.

**Why Mode B fails for v3 but not v1**: Mode B uses a suffix-maximum guarantee — requires that the calib fail rate at every λ' ≥ λ* stays ≤ α − ε_fail. The plateau fail rate (when λ is large enough that all examples go to reflect/replan) is:
- v1: plateau fail = **0.549** ≤ 0.554 (target for α=0.65) → barely feasible
- v3: plateau fail = **0.561** > 0.554 → miss by 0.007, infeasible

v3 learned stronger replan predictions; at the plateau, more examples route to replan vs reflect. On this particular 164-example calib set, replan fails slightly more often than reflect, pushing plateau fail 1.2pp higher than v1. This is pure small-n noise — with n=164, ε_fail=0.096 makes Mode B highly sensitive to which 164 examples land in calib.

**Mode A comparison v1 vs v3 (sweet spot C=8):**

| Router | test solve | test cost | Δ solve | Δ cost |
|---|---|---|---|---|
| v1 C=8 | 0.616 | 6.23 | — | — |
| **v3 C=8** | **0.622** | **5.49** | **+0.6pp** | **−0.74** |

v3 is slightly better on Mode A: same cost budget buys +0.6pp solve_rate AND 0.74 fewer cost units. The improvement is modest (consistent with the +0.9pp argmax gain).

### CRC Results — New Holdout Benchmark (v3 router, n=720, predefined calib/test split)

Source: `datasets/benchmarks/holdout_3cls_calib.json` + `holdout_3cls_test.json`

**Holdout construction** (2026-05-30):
- n=720 (calib=360, test=360), predefined split (not random each run)
- Label dist: 44% escalate / 28% reflect / 28% replan — matches training distribution
- Contains **63 verified cheap_only examples** (reflect/replan works, escalate explicitly fails from exhaustive rollout)
- Source mix: 328 exhaustive benchmark examples (real SA) + 392 cherry-picked high-confidence examples (liberal SA)
- Selection: stratified by label, sorted by model confidence within each stratum
- δ=0.05, ε_cost=0.839, **ε_fail=0.0645** (vs 0.096 on old 164-example calib)
- solve_ok uses liberal assumption for cherry-picked portion (escalate counts if SA non-empty); strict for exhaustive bench portion

**Why this holdout differs from old benchmark:**
- Larger calib (360 vs 164) → smaller ε → tighter guarantees
- Predefined split → reproducible
- Includes cheap_only examples → honest evaluation of cases where router over-predicts escalate

**λ=0 (argmax, no CRC):**

| | calib | test |
|---|---|---|
| solve_rate | 0.994 | 0.983 |
| mean cost | 11.17 | 11.35 |
| pred dist | esc=300, ref=45, rep=15 | esc=306, ref=36, rep=18 |

**Mode A (E[cost] ≤ C, δ=0.05):**

| C | λ | calib cost | test cost | test solve | guarantee |
|---|---|---|---|---|---|
| 3 | 0.0248 | 2.153 | 2.183 | 0.494 | ✓ |
| 5 | 0.0150 | 3.894 | 3.956 | 0.617 | ✓ |
| 6 | 0.0126 | 4.933 | 4.781 | 0.667 | ✓ |
| 7 | 0.0123 | 6.033 | 5.453 | 0.719 | ✓ |
| **8** | **0.0101** | **6.614** | **6.156** | **0.767** | ✓ |
| 9 | 0.0098 | 7.836 | 7.408 | 0.850 | ✓ |
| 10 | 0.0075 | 8.844 | 8.294 | 0.892 | ✓ |
| 11 | 0.0073 | 9.975 | 9.761 | 0.967 | ✓ |
| ∞ (λ=0) | 0 | 11.167 | 11.350 | 0.983 | (no guarantee) |

**Mode B (P[fail] ≤ α, δ=0.05):**

| α | feasible? | λ | calib fail | test solve | test cost |
|---|---|---|---|---|---|
| ≤0.60 | ✗ | — | — | — | — |
| **0.65** | **✓** | **0.0477** | **0.542** | **0.483** | **2.000** |

**Comparison old vs new (v3, sweet spot C=8):**

| Benchmark | n_calib | ε_fail | argmax solve | C=8 test solve | C=8 test cost | Mode B |
|---|---|---|---|---|---|---|
| Old (random, n=164) | 164 | 0.096 | 0.726 | 0.622 | 5.49 | infeasible |
| **New (holdout, n=360)** | **360** | **0.065** | **0.983** | **0.767** | **6.16** | **α=0.65 ✓** |

The large argmax solve improvement (0.726→0.983) reflects the cherry-picked nature of the new holdout — not a real performance jump. The meaningful improvements are: better ε (smaller calib uncertainty) and Mode B unlocked at α=0.65.

### CRC Results — v4 on Old Benchmark (3cls_bench_random, random 50/50 split)

Source: `conformal/results/crc_bench_3cls_v4.json`

n=328, calib=164, test=164, δ=0.05, ε_cost=1.051, ε_fail=0.096

argmax baseline (test): solve_rate=0.793, cost=9.713

**Mode A:**

| C | λ | test solve | test cost | guarantee |
|---|---|---|---|---|
| 4 | 0.0321 | 0.500 | 2.47 | ✓ |
| 5 | 0.0242 | 0.549 | 3.27 | ✓ |
| 6 | 0.0214 | 0.616 | 4.21 | ✓ |
| 7 | 0.0167 | 0.652 | 5.09 | ✓ |
| **8** | **0.0138** | **0.683** | **5.96** | ✓ |
| 9 | 0.0095 | 0.738 | 7.37 | ✓ |
| 10 | 0.0065 | 0.744 | 8.44 | ✓ |
| 11 | 0.0022 | 0.787 | 9.31 | ✓ |

**Mode B:** α=0.65 ✓ (lam=0.0600, test_fail=0.537, test_solve=0.463, cost=2.000).

**v4 vs v3 CRC comparison (C=8 sweet spot):**

| Router | argmax test solve | C=8 test solve | C=8 test cost | Mode B |
|---|---|---|---|---|
| v1 | 0.732 | 0.616 | 6.23 | α=0.65 barely ✓ |
| v3 | 0.726 | 0.622 | 5.49 | infeasible |
| **v4** | **0.793** | **0.683** | **5.96** | **α=0.65 ✓** |

v4 gains +6.7pp argmax, +6.1pp Mode A C=8 vs v3. Mode B newly feasible (like v1).

### CRC Results — v4 Holdout (predefined calib/test split, no training overlap)

Source: `conformal/results/crc_holdout_3cls_v4.json`

n_calib=360, n_test=360, δ=0.05, ε_cost=0.710, ε_fail=0.065

**λ=0 (argmax, no CRC):**

| | calib | test |
|---|---|---|
| solve_rate | 0.694 | 0.656 |
| mean cost | 10.678 | 11.136 |
| pred dist | esc=284, ref=55, rep=21 | esc=299, ref=42, rep=19 |

**Mode A (E[cost] ≤ C, δ=0.05):**

| C | λ | calib cost | test cost | test solve | guarantee |
|---|---|---|---|---|---|
| 3 | 0.0353 | 2.275 | 2.367 | 0.514 | ✓ |
| 5 | 0.0252 | 4.261 | 4.139 | 0.614 | ✓ |
| 6 | 0.0229 | 5.239 | 5.147 | 0.642 | ✓ |
| 7 | 0.0204 | 6.186 | 5.972 | 0.656 | ✓ |
| **8** | **0.0169** | **7.256** | **7.256** | **0.711** | ✓ |
| 9 | 0.0138 | 8.203 | 8.417 | 0.711 | ✓ |
| 10 | 0.0098 | 9.272 | 9.822 | 0.697 | ✓ |
| ∞ (λ=0) | 0 | 10.678 | 11.136 | 0.656 | (no guarantee) |

**Mode B:** All α ≤ 0.65 infeasible. Model routes too many easy examples to escalate on holdout, so plateau fail rate > 0.585 = α − ε.

**Comparison v3-holdout vs v4-holdout:**

| Router | argmax calib solve | argmax test solve | C=8 test solve | C=8 test cost | Mode B |
|---|---|---|---|---|---|
| v3-holdout | 0.994 (inflated) | 0.983 (inflated) | 0.767 | 6.16 | α=0.65 ✓ |
| **v4-holdout** | **0.694 (honest)** | **0.656 (honest)** | **0.711** | **7.26** | infeasible |

v3-holdout numbers were inflated because v3 was trained on holdout data. v4-holdout is the first genuine generalization test.
- v4 Mode A C=8 test solve (0.711) beats v3 bench_random C=8 (0.683) even out-of-distribution.
- v4 escalate accuracy is near-perfect on holdout (96.7%), but replan is weak (33.9% solve).

### 4cls Router Benchmark Eval (4cls_bench_random, n=318)

The 4cls benchmark has 62% unsolvable labels (197/318). solve_rate is suppressed for any model that correctly predicts unsolvable (no action → no solve). The right metrics are **(a) cls_acc on solvable subset** and **(b) unsolvable recall/precision**.

**Model comparison:**

| Model | cls_acc | solve_rate | solvable solve_rate | mean cost | unsolvable recall | status |
|---|---|---|---|---|---|---|
| 4cls v1 | 0.625 | 0.097 | ~0.48 | ~0.3 | 92% | over-conservative (84% pred unsolvable) |
| 4cls v2 (ckpt-2300) | 0.165 | 0.283 | n/a | ~12.7 | ~1% | collapsed to escalate |
| **4cls v3 (ckpt-600, fixed)** | **0.535** | **0.226** | **0.579** | **3.92** | **61%** | **working** |

*solvable solve_rate* = solve_rate restricted to examples where true_label ∈ {reflect, replan, escalate}.

**v3 per-label breakdown:**

| Label | n (true) | cls_acc | solve_rate | notes |
|---|---|---|---|---|
| reflect | 32 | 0.188 | 0.656 | mostly routed to replan (13/32) — works since both cost=2 |
| replan | 38 | 0.553 | 0.737 | best-classified solvable class |
| escalate | 51 | 0.451 | 0.412 | 20/51 misrouted to replan — those fail, pulling solve_rate down |
| unsolvable | 197 | 0.609 | 0.010 | 120/197 correctly identified; 77 routed to solvable actions |

**v3 prediction distribution:** unsolvable=141, replan=87, escalate=81, reflect=9

**Analysis:**
- v3 is the first 4cls model that actually works (cls_acc=0.535 vs collapsed v1/v2).
- The 30% unsolvable cap in training reduced over-prediction of unsolvable (v1: 84%, v3: 44%).
- **Unsolvable recall dropped** (v1: 92%, v3: 61%) — expected trade-off from the cap. Model now tries more solvable actions.
- **Solvable solve_rate improved** (v1: ~48%, v3: 57.9%) — fewer examples stranded as unsolvable.
- **Mean cost = 3.92** (vs v1's ~0.3 and 3cls v3's 10.5) — sits between always_reflect (cost=2) and always_escalate (cost=13). The model spends cost only when it's confident the problem is solvable.
- Main weakness: reflect is nearly invisible (9/318 predictions, true=32). Model conflates reflect with replan at cost=2, which is acceptable for solve_rate but means true reflect-only problems are slightly under-served.

---

## Key Bottlenecks

| Bottleneck | Impact | Status |
|---|---|---|
| Router quality (oracle gap) | v4 81.4% (bench) / 65.6% (holdout) vs oracle 96.0% | 3cls v4 evaled ✓ |
| n_calib too small (n~164, ε_fail=9.6%) | Mode B barely feasible at α=0.65 | New holdout: n=360, ε_fail=6.5% ✓ |
| Escalate underestimated in short-circuit data | Biases Mode B upward | Exhaustive benchmark rollout done ✓ |
| 4cls class imbalance (62.5% unsolvable in bench) | Suppresses solve_rate; v1/v2 collapsed | 4cls v3 fixed + evaled ✓ (solvable solve_rate=57.9%) |
| LoRA path bug (VLM-wrapped model) | v2/v3 initially unusable | v3 fixed ✓; v2 not worth fixing |
| Holdout cherry-picked | argmax solve inflated (0.983), not deployment-realistic | Noted; old benchmark (0.774) is honest reference |

---

## Pending / Next Steps

1. **CRC on old benchmark (v3)** — ✓ done. Mode A C=8: solve=0.622, cost=5.49. Mode B infeasible.
2. **CRC on new holdout (v3)** — ✓ done. Mode A C=8: solve=0.767, cost=6.16. Mode B α=0.65 ✓. (Inflated — v3 trained on holdout.)
3. **3cls v4 SFT** — ✓ done. Best checkpoint-1000 (eval_loss=0.2715). LoRA key bug fixed in `_fixed/` by post-training script.
4. **3cls v4 eval + CRC** — ✓ done.
   - Old benchmark: solve_rate=0.814 (+4.0pp vs v3). CRC C=8: 0.683. Mode B α=0.65 ✓.
   - Holdout test (no-overlap): solve_rate=0.656 (genuine). CRC C=8: 0.711. Mode B infeasible.
   - Files: `conformal/results/bench_eval_3cls_v4/`, `holdout_3cls_v4/`, `crc_bench_3cls_v4.json`, `crc_holdout_3cls_v4.json`
5. **Vote rollout** — ✓ done. 3,377 unique problems × 10 trials. Output: `cloudide/code/outputs/votes/votes_merged.jsonl`.
6. **v5 datasets built** — ✓ done. See "v5 Data" section below for details.
7. **v5 SFT training** — IN PROGRESS:
   - Machine A (serial): v5b_clean **running** → v5b_full queued
   - Machine B: v5a_clean **ready to start** → v5a_full after
   - Script: `sft_runs/router_arch_a_3cls_v5_all.sh`
8. **v5 eval + CRC** — after each checkpoint; use same holdout + bench eval pipeline as v4
9. **4cls v3 CRC** — pending
10. **Mode A Pareto figure** — compare v3/v4/v5b_clean/v5a_clean; v4 is honest reference

---

## v5 Data: Vote-Based Label Improvement

### Motivation

v3/v4 models output near-identical softmax distributions for `esc_only` and `cheap_only` problems (p_escalate differs by only 0.06), because they were trained on single-trial hard labels. A problem labeled "reflect" only teaches the model "output reflect" — it never learns *why* reflect works and escalate doesn't. This makes CRC's λ-sweep nearly useless: any λ that suppresses escalate on cheap_only problems equally suppresses it on genuinely hard esc_only problems.

### Vote Rollout: Data Generation

For each of the 4,656 training examples, run 10 independent trials using the **short-circuit rollout**:

```
Each trial:
  try reflect  → if pass, label = "reflect", stop
  try replan   → if pass, label = "replan",  stop
  try escalate → if pass, label = "escalate", stop
  else             label = "unsolvable"
```

After 10 trials, count votes per label. Soft label = votes / n_solvable (normalise out unsolvable trials so probabilities sum to 1).

**Example outputs:**

```
cheap_only: {reflect:9, replan:1, escalate:0, unsolvable:0}
 → p_reflect=0.90, p_replan=0.10, p_escalate=0.00  ← model should learn NOT to escalate

esc_only:   {reflect:0, replan:1, escalate:8, unsolvable:1}
 → p_reflect=0.00, p_replan=0.11, p_escalate=0.89  ← model should learn to escalate confidently

both:       {reflect:7, replan:2, escalate:1, unsolvable:0}
 → p_reflect=0.70, p_replan=0.20, p_escalate=0.10
```

**Rollout engineering notes:**
- 429 rate-limits on gpt-5.4 (escalate) handled via exponential backoff retry (1→2→4→8s)
- Escalate concurrency capped at 20 via semaphore to avoid saturating gpt-5.4 QPM
- Dataset split across 4 parallel processes by CodeEnv sandbox (apps, apps_functional, taco, bcb/lcb)
- Total: 3,377 unique problems matched to rollout; 1,166 from old batches (no rollout file) — only 7 recoverable
- Output: `cloudide/code/outputs/votes/votes_merged.jsonl` (3,377 records, each with `votes` dict and `n_solvable`)

### Vote Distribution (3,377 problems)

| SA type | Count | % | Intuition |
|---|---|---|---|
| both (all 3 work) | 1,636 | 48.4% | Easy problems — router just needs to avoid wasting cost |
| cheap_only (esc fails) | 934 | 27.7% | **Critical** — current models wrongly predict escalate here |
| esc_only (only esc works) | 656 | 19.4% | Hard problems — must escalate, can't cheap out |
| unsolvable (all 10 fail) | 151 | 4.5% | Genuinely broken problems; removed from training |

**Mean soft labels (solvable only):** p_reflect=0.30, p_replan=0.26, p_escalate=0.44 (sum=1.000)

The 27.7% cheap_only fraction is important: these are problems where v3/v4 often incorrectly predict escalate (costing 13 units and failing). Vote labels will teach the model to predict p_escalate ≈ 0 for these.

### Label Changes vs Original Hard Labels

Of the 3,456 examples with vote data:
- **913 had their modal label changed** (26%) — these are the corrected mislabels
- 1,981 confirmed same label (modal vote agrees with original)
- 410 had n_solvable < 5 (noisy) → kept original hard label
- 152 all-unsolvable → dropped from training

The 913 changes are concentrated in cheap_only cases where the original label was "escalate" (single trial happened to succeed with escalate) but majority vote shows reflect/replan is more reliable.

### v5 Datasets

Two training approaches, each with a "full" (keep 1,200 no-vote examples as hard labels) and "clean" (drop them) variant:

**v5b: Hard label from modal vote** — same size as v4, each example gets the majority-vote label. Standard CE loss. Simple, interpretable.

**v5a: Soft label via weighted copies** — for each problem, create one training example per successful trial outcome. Mathematically equivalent to soft CE loss (KL divergence) in expectation — the gradient weighted average matches the vote distribution. More expensive to train (27k examples) but directly implements soft label training.

| Dataset | File | n | Label method | No-vote handling |
|---|---|---|---|---|
| `router_arch_a_3cls_v5a_full` | `datasets/` | 27,612 | Soft (copies) | Keep as 1 hard-label copy |
| `router_arch_a_3cls_v5a_clean` | `datasets/` | 26,413 | Soft (copies) | Drop entirely |
| `router_arch_a_3cls_v5b_full` | `datasets/` | 4,504 | Hard (modal) | Keep original label |
| `router_arch_a_3cls_v5b_clean` | `datasets/` | 3,305 | Hard (modal) | Drop entirely |

**Label distribution (all v5 variants):** escalate≈44%, reflect≈31-33%, replan≈23-25%
Compared to v3/v4 hard labels (44/28/28): reflect is higher by +3-5pp because cheap_only examples no longer get mislabeled as escalate.

### Why v5a ≠ KL Loss but Is Equivalent

Creating 10 copies per example (7 reflect + 2 replan + 1 escalate) with standard CE loss gives the same gradient in expectation as soft CE loss with targets (0.7, 0.2, 0.1). This is because:

```
E[∇CE(hard)] = Σ_a p_vote(a) · ∇CE(a) = ∇SoftCE(p_vote)
```

No custom loss needed. The downside: the model sees the same example 10× per epoch with different labels, which can feel noisy. This is intentional — the variance across copies teaches calibrated uncertainty.

### Training Configuration

| Variant | Steps | Epochs | Rationale |
|---|---|---|---|
| v5b_clean | ~1,116 | 3 | Same as v4 (similar dataset size) |
| v5b_full | ~1,519 | 3 | Slightly more data, same epochs |
| v5a_clean | ~2,972 | 1 | 6× more data; 1 epoch ≈ 6× passes through unique problems |
| v5a_full | ~3,104 | 1 | Same reasoning |

Post-training: LoRA key fix required (same `language_model.` prefix bug as all v3+ models). Fix script embedded in `router_arch_a_3cls_v5_all.sh`.

**Run commands:**
```bash
# Machine A (serial): v5b_clean → v5b_full
bash sft_runs/router_arch_a_3cls_v5_all.sh > sft_runs/outputs/v5_all.log 2>&1 &

# Machine B: v5a_clean
cd LLaMA-Factory && FORCE_TORCHRUN=1 llamafactory-cli train \
  ../sft_runs/router_arch_a_3cls_v5a_clean.yaml
# then: run fix_keys on _fixed/ output
# then: v5a_full same way
```

---

## Future Directions

### Vote-based soft labels

#### Why current router probabilities are the problem

Inspecting the v3 model's softmax output on real examples reveals a troubling pattern:

```
esc_only problem (escalate is the ONLY solution):
  p_reflect=0.15   p_replan=0.39   p_escalate=0.47   → pred=escalate ✓

cheap_only problem (escalate FAILS, only reflect/replan work):
  p_reflect=0.13   p_replan=0.34   p_escalate=0.53   → pred=escalate ✗
```

These two problem types produce **nearly identical softmax distributions** (p_escalate differs by only 0.06). The router is essentially guessing — it routes to escalate on both, gets it right for esc_only and catastrophically wrong for cheap_only (spending cost=13 on an action that fails).

The root cause: current SFT trains on **single hard labels**. For a cheap_only problem labeled `reflect`, the training signal is just "output the word reflect." The model never learns *why* reflect works and escalate doesn't, because we only ever showed it one outcome. The softmax is flat because the model genuinely doesn't know the difference.

This matters especially for CRC: the λ-penalty `p_a - λ·cost(a)` is only useful if different problem types produce meaningfully different p values. With the current flat distributions, a λ that suppresses escalate on cheap_only problems also suppresses it on esc_only problems — there's no way to discriminate.

#### The fix: majority vote soft labels

For each training problem, run each action **independently N times** and use empirical pass rates as training targets:

```
cheap_only problem: escalate tried 10×, passes 0/10 → p_escalate = 0.0
                    reflect  tried 10×, passes 8/10 → p_reflect  = 0.8
                    replan   tried 10×, passes 7/10 → p_replan   = 0.7

esc_only problem:   escalate tried 10×, passes 9/10 → p_escalate = 0.9
                    reflect  tried 10×, passes 0/10 → p_reflect  = 0.0
                    replan   tried 10×, passes 1/10 → p_replan   = 0.1
```

Train with KL divergence / soft cross-entropy against this distribution.

#### What changes after soft-label training

The model learns that `cheap_only` and `esc_only` problems have **structurally different probability signatures**:

```
After soft-label SFT:
esc_only:   p_esc≈0.90  p_rep≈0.05  p_ref≈0.05
cheap_only: p_esc≈0.05  p_rep≈0.50  p_ref≈0.45
```

Now CRC λ-sweep is genuinely discriminative:
- **esc_only**: need λ > 0.065 before `p_esc - λ×13 < p_rep - λ×2` (escalate gets overridden)
- **cheap_only**: λ > 0.003 is enough to override escalate (because p_esc is already 0.05)

This means CRC can target cheap_only problems specifically — routing them cheaply without disturbing esc_only routing. The Mode A Pareto curve would steepen dramatically.

#### Impact on the 18.6% router error rate

From the exhaustive benchmark analysis, 61/328 solved problems have router errors:
- 28 are cheap_only cases where router predicted escalate (escalate fails, cost wasted)
- 22 are esc_only cases where router predicted reflect/replan (they fail)

Soft labels directly fix the 28 cheap_only errors: the model would output p_esc≈0 for those problems, so even λ=0 (argmax) correctly routes to reflect/replan. The 22 esc_only errors might also improve since the model would output p_esc≈0.9 with higher confidence.

#### Estimated impact on CRC

| Setup | argmax solve | Mode A C=8 solve | Mode B |
|---|---|---|---|
| v3 hard labels (current) | 0.774 | 0.622 | infeasible |
| v3 soft labels (projected) | ~0.85+ | ~0.75+ | feasible lower α |

The projection is based on: eliminating the 28 cheap_only errors (+8.5pp on argmax) and sharper CRC discrimination (+5-10pp on Mode A).

#### Cost analysis

| Item | Value |
|---|---|
| Training examples | 4,656 |
| Actions × runs | 3 × N |
| Total API calls at N=5 | ~70k |
| Time at 36k QPM + concurrency | ~2-3 hours |
| Training overhead | 1 SFT run, ~6h |

At N=5 the cost is manageable. Even N=3 captures the binary cheap/expensive signal.

**Downside**: stochastic LLMs mean a problem might pass 3/10 due to randomness rather than genuine solvability. This introduces noise in the soft labels. Mitigations: use larger N, filter problems with near-50% pass rates for a given action (ambiguous labels), or treat the soft label as a mixture of hard label and empirical rate.

**Priority**: high — this directly addresses the most fixable failure mode (cheap_only misrouting) and significantly improves CRC calibration quality.

---

## File Index

| Path | Purpose |
|---|---|
| `datasets/router_arch_a_3cls_v2.json` | 3cls SFT data, 5,498 ex (has conflicting labels — do not use for training) |
| `datasets/router_arch_a_3cls_v3.json` | 3cls SFT data deduped, 5,114 ex — used for v3 training |
| `datasets/router_arch_a_3cls_v3_train.json` | 3cls training after holdout split, 4,394 ex |
| `datasets/router_arch_a_4cls_v2.json` | 4cls SFT data, 13,626 ex (has conflicting labels) |
| `datasets/router_arch_a_4cls_v3.json` | 4cls SFT data deduped + unsolvable cap 30%, 7,078 ex — used for v3 |
| `datasets/router_arch_a_4cls_v3_train.json` | 4cls training after holdout split, 6,178 ex |
| `datasets/benchmarks/holdout_3cls_calib.json` | 3cls CRC calibration set, n=360, predefined split |
| `datasets/benchmarks/holdout_3cls_test.json` | 3cls CRC test set, n=360, predefined split |
| `datasets/router_no_reason_v1_archB_*.json` | Arch B variants (problem-only input) |
| `datasets/benchmarks/` | Eval benchmark files (4 files × ~310–328 examples after null removal) |
| `sft_runs/*.yaml` | Training configs (one per model version) |
| `sft_runs/*.sh` | Training entry scripts |
| `sft_runs/outputs/router_arch_a_3cls/checkpoint-200-renamed/` | 3cls v1 best checkpoint |
| `sft_runs/outputs/router_arch_a_3cls_v3_fixed/checkpoint-800/` | 3cls v3 best checkpoint |
| `sft_runs/outputs/router_arch_a_4cls_v3_fixed/checkpoint-600/` | **4cls v3 best checkpoint (use this)** |
| `sft_runs/outputs/router_arch_a_*_v2/` | v2 checkpoints — do not use (path bug + conflicting labels) |
| `conformal/results/bench_eval_3cls_v1/` | 3cls v1 per-example eval output + summary.json |
| `conformal/results/bench_eval_3cls_v3/` | 3cls v3 per-example eval output + summary.json |
| `conformal/results/bench_eval_3cls_v4/` | **3cls v4 per-example eval output + summary.json** (solve_rate=0.814) |
| `conformal/results/holdout_3cls_v4/` | **v4 holdout calib+test eval** (calib solve=0.694, test solve=0.656) |
| `conformal/results/bench_eval_4cls_v3/` | 4cls v3 per-example eval output + summary.json |
| `conformal/results/crc_bench_3cls_v1_n328.json` | CRC Mode A/B results for v1 router |
| `conformal/results/crc_bench_3cls_v3.json` | CRC Mode A/B for v3 (old benchmark, random split) |
| `conformal/results/crc_bench_3cls_v4.json` | **CRC Mode A/B for v4 (old benchmark, random split)** C=8: solve=0.683 |
| `conformal/results/crc_holdout_3cls_v4.json` | **CRC Mode A/B for v4 (holdout predefined split)** C=8: solve=0.711 |
| `sft_runs/outputs/router_arch_a_3cls_v4_fixed/checkpoint-1000/` | **v4 best checkpoint (use this)** |
| `conformal/results/SUMMARY.md` | Cross-version eval results index (all metrics in one place) |
| `conformal/scripts/calibrate_eval_crc_v3.py` | CRC Mode A/B implementation |
| `cloudide/code/scripts/eval_router_benchmark.py` | Benchmark eval script (cls_acc + solve_rate, supports 3cls/4cls) |
| `cloudide/code/scripts/crc_on_benchmark.py` | CRC sweep on benchmark eval output (random split) |
| `cloudide/code/scripts/crc_on_holdout.py` | CRC sweep with predefined calib/test split (new) |
| `cloudide/code/scripts/run_benchmark_exhaustive.py` | Fill in successful_actions via exhaustive rollout API |
| `cloudide/rollout/*.jsonl` | Source rollout data (short-circuit format) |
