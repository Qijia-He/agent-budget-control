# Router Eval Summary

Last updated: 2026-05-29

This document maps every eval result to its source file and explains what each number means.

---

## Quick reference: where to find results

| Result | File |
|--------|------|
| 3cls v1 benchmark eval (solve_rate) | `bench_eval_3cls_v1/summary.json` |
| 3cls v2 benchmark eval (solve_rate) | `bench_eval_3cls_v2/summary.json` |
| 4cls v1 benchmark eval (solve_rate) | `bench_eval_4cls_v1/summary.json` |
| 4cls v2 benchmark eval (solve_rate) | `bench_eval_4cls_v2/summary.json` |
| 3cls v1 CRC on benchmark | `crc_bench_3cls_v1.json` |

---

## Metric note

There are **two different eval metrics** used at different stages:

- **dec_acc** (old): fraction of problems where argmax prediction == ground-truth label. Used in early CRC runs (eval.json, eval_nano.json, crc_v3_*.json). Lower bound — doesn't credit the router when a non-optimal action also works.
- **solve_rate** (new): fraction of problems where argmax prediction ∈ successful_actions. Requires the exhaustive benchmark rollout (completed 2026-05-27). Correct metric. Numbers are ~20pp higher than dec_acc.

Do not compare dec_acc and solve_rate directly.

---

## Argmax baseline comparison (no conformal)

### Old metric (dec_acc, n≈149–159, test split from old calib set)

| Router | dec_acc | mean cost |
|--------|---------|-----------|
| GPT-nano argmax | 0.342 | 2.0 |
| GPT-mini argmax | 0.340 | 2.0 |
| Qwen3.5-4B base (no SFT) | 0.358 | 2.0 |
| **3cls v1 SFT argmax** | **0.522** | 10.6 |

Source: `eval_nano.json`, `eval_mini.json`, `eval_qwenbase.json`, `eval.json`

GPT/Qwen base routers are barely above the always-reflect baseline; SFT gives a large jump (+16pp).

### New metric (solve_rate, benchmark eval)

#### Benchmark 文件现状（2026-05-29 更新）

null SA 例子已全部删除（原因：无 rollout 数据且无法溯源数据集）。stratified 文件在清理过程中丢失，目前仅保留 random 版本。

| 文件 | n | n_with_sa | unsolvable | 说明 |
|------|---|-----------|-----------|------|
| `3cls_bench_random.json` | 328 | 315 | 13 | 主力 eval，无 null ✅ |
| `4cls_bench_random.json` | 318 | 122 | 196 | unsolvable 占 62%，solve_rate 受抑 |
| `archB_4cls_bench_random.json` | 302 | 297 | 5 | 质量最好 |
| `archB_5cls_bench_random.json` | 310 | 207 | 103 | — |

#### 3cls 模型 eval 结果

| Model | Bench | cls_acc | solve_rate | n_sa |
|-------|-------|---------|------------|------|
| 3cls v1 (ckpt-200) | random | 0.547 | 0.765 | 328 |
| 3cls v2 (ckpt-1000) | random | 0.268 | 0.283 | 328 |
| **3cls v3 (ckpt-800, fixed)** | **random** | **0.546** | **0.774** | **328** |

注：3cls v2 路径 bug 导致 adapter 完全无效（见 collapse log）。3cls v3 有相同的路径 bug（训练时模型有额外的 `language_model` wrapper），修复后 solve_rate=0.774，略优于 v1。

Source: `bench_eval_3cls_v1/summary.json`, `bench_eval_3cls_v2/summary.json`, `bench_eval_3cls_v3/summary.json`

#### 4cls 模型 eval 结果

> **注意**：4cls benchmark 中 unsolvable 占 62%（197/318）。模型正确预测 unsolvable 会得 cls_acc 加分，但 solve_rate=0（没有执行动作）。因此 solve_rate 对 4cls 模型不公平，应以 solvable 子集的 cls_acc 为主要指标。

| Model | cls_acc | solve_rate | n_sa | 状态 | 主要预测 |
|-------|---------|------------|------|------|---------|
| 4cls v1 | 0.625 | 0.097 | 318→122 | 过度保守 | unsolvable 84% |
| 4cls v2 (ckpt-2300) | 0.165 | 0.283 | 318→122 | 坍塌 | escalate 98.5% |
| 4cls v3 | — | — | — | ⏳ 待训练 | — |

Source: `bench_eval_4cls_v1/`, `bench_eval_4cls_v2/summary.json`

#### Action-level baseline（3cls benchmark，n=625 with SA）

| Strategy | solve_rate | mean cost |
|----------|------------|-----------|
| always_reflect | 0.270 | 2 |
| always_replan | 0.413 | 2 |
| always_escalate | 0.707 | 13 |
| random | 0.442 | ~5.7 |
| **3cls v1 argmax** | **0.735** | 10.5 |
| oracle (perfect routing) | 0.955 | 6.7 |

**Oracle 说明**：oracle 不是一个模型，而是数据本身决定的上界：

```
oracle solve_rate = 有 non-empty successful_actions 的题数
                   ─────────────────────────────────────
                   所有有 SA 记录的题数（含 empty）
```

benchmark 里约 4.5% 的题是真正无解的（reflect / replan / escalate 全部失败），即使路由完全正确也无法解决。cost=6.7 表示完美路由下的平均花费：约 43% 的题只有 escalate 能解（cost=13），其余用 reflect/replan 就够（cost=2）。

3cls v1 argmax（0.735）与 oracle（0.955）之间的 **0.22 gap** 是路由质量还能提升的空间上界。

---

## Conformal (CRC) results — 3cls v1 on new benchmark

Using `crc_bench_3cls_v1.json` (n=625 with SA, calib=312, test=313, δ=0.05).

### Mode A — cost budget (E[cost] ≤ C)

| C | test solve_rate | test cost | guarantee held? |
|---|-----------------|-----------|-----------------|
| 5 | 0.562 | 4.32 | ✓ |
| 7 | 0.639 | 6.36 | ✓ |
| **8** | **0.665** | **7.31** | **✓ sweet spot** |
| 9 | 0.690 | 8.22 | ✓ |
| 10 | 0.716 | 9.24 | ✓ |
| argmax (no CRC) | 0.735 | 10.5 | — |

### Mode B — fail budget (P[fail] ≤ α)

| α | feasible? | test solve_rate | test cost |
|---|-----------|-----------------|-----------|
| 0.60 | ✗ | — | — |
| **0.65** | **✓** | **0.503** | **2.0** |

Mode B is bottlenecked by calib set size (n=312, ε_fail=0.069). To unlock α<0.65, need n≥500.

Source: `crc_bench_3cls_v1.json`

---

## Model collapse log

| Model | Best ckpt | Behavior | Root cause | Usable? |
|-------|-----------|----------|------------|---------|
| 3cls v2 (ckpt-1000) | eval_loss=0.2851 | Predicts `reflect` 90%+ — but actually a **path bug**: adapter key had extra `language_model.` prefix, weights never applied | Training was on a VLM-wrapped model; PEFT silently skipped mismatched keys | No (retrain needed) |
| 3cls v3 (ckpt-800) | eval_loss=0.2814 | Same path bug — **fixed** by renaming safetensors keys; solve_rate=0.774 after fix | Same VLM wrapper issue | **Yes (use fixed version)** |
| 4cls v1 | — | Over-predicts `unsolvable` (336/400 = 84%, true=250) | Class imbalance (62.5% unsolvable in training) | Partial |
| 4cls v2 (ckpt-2300) | eval_loss=0.2397 | Predicts `escalate` for 394/400 | Same class imbalance, collapses opposite direction | No |

**Path bug explanation**: v2 and v3 were trained when LlamaFactory loaded `Qwen/Qwen3.5-4B` with an internal `language_model` wrapper (possibly due to a `trust_remote_code` path). The saved LoRA keys had the form `base_model.model.model.language_model.layers.*` instead of `base_model.model.model.layers.*`. PEFT found no matching modules and silently created orphan layers — adapter had zero effect. Fix: rename keys in safetensors to remove `language_model.`. Applied to all v3 checkpoints in `router_arch_a_3cls_v3_fixed/`.

Fix for future runs: verify adapter effect on a small sample before running full eval (compare a few logits base vs. adapted — they must differ).

---

## Checkpoints

| Model | Path | Notes |
|-------|------|-------|
| 3cls v1 | `sft_runs/outputs/router_arch_a_3cls/checkpoint-200-renamed` | solve_rate=0.765, good baseline |
| **3cls v3 (fixed)** | **`sft_runs/outputs/router_arch_a_3cls_v3_fixed/checkpoint-800`** | **solve_rate=0.774, best available — use this** |
| 4cls v1 | `sft_runs/outputs/router_arch_a_4cls-renamed` | Over-conservative on unsolvable; needs retraining |
| 3cls v2 | `sft_runs/outputs/router_arch_a_3cls_v2/checkpoint-1000` | Path bug, adapter has zero effect, do not use |
| 4cls v2 | `sft_runs/outputs/router_arch_a_4cls_v2/checkpoint-2300` | Class collapse, do not use |

---

## Next steps

1. **3cls v3 eval** — 训练完成后跑 `eval_router_benchmark.py` on `3cls_bench_random.json`，对比 v1
2. **4cls v3 训练** — 用 `router_no_reason_v1_4cls_deduped.json`（去重 + unsolvable cap 30%）
3. **Expand calib set** — target n≥500 to get ε_fail<0.05 and unlock Mode B at α=0.60
4. **GPT router on new benchmark** — run GPT-nano/mini on `3cls_bench_random.json` to get solve_rate for fair comparison
