# Router + CRC Progress

Last updated: 2026-05-29

---

## 1. SFT 训练数据版本

### 数据集命名说明

这里的 v1/v2/v3 指的是 **SFT 训练数据版本**，与下方 CRC 方法论版本（同名但无关）区分开。

所有数据通过 `cloudide/code/scripts/build_sft_router_v1.py` 从 rollout 结果生成，字段格式统一：
```
instruction: 系统提示（cost-aware router 角色说明）
input: "Problem:\n<题目>\n\nInitial attempt verdict: <fail|timeout|compile_error>\nInitial attempt stderr:\n<错误信息>"
output: 单词标签（reflect / replan / escalate / unsolvable）
```

### 3cls 系列（标签：reflect / replan / escalate）

| 版本 | 数据集文件 | 样本数 | 标签分布 | 问题 | 状态 |
|------|-----------|--------|----------|------|------|
| **v1** | `router_no_reason_v1_3cls.json`（前1583条）| 1583 | escalate 48%, replan 28%, reflect 24% | — | ✅ 最佳，checkpoint-200 |
| **v2** | `router_no_reason_v1_3cls.json`（全5498条）| 5498 | escalate 45%, reflect 28%, replan 28% | 有438道题与v1重复，其中166条标签冲突（38%）；模型被矛盾梯度拖垮，坍塌为 always_reflect | ❌ 坍塌 |
| **v3** | `router_no_reason_v1_3cls_deduped.json` | 5114 | escalate 44%, reflect 28%, replan 28% | v2 按 problem key 去重（keep-first），消除166条标签冲突 | ⏳ **训练中**（2026-05-29）|

**v2 坍塌根因**：`build_sft_router_v1.py` 以 `(dataset_tag, problem_id)` 为 key 去重，同一道题若出现在不同 rollout 文件中会被处理两次。新旧 rollout 产生的 `successful_actions` 不同 → `cheapest_action` 标签不同 → 同一输入在训练集里有两种标签，梯度互相抵消，模型退化为保守的 always_reflect。

### 4cls 系列（标签：reflect / replan / escalate / unsolvable）

| 版本 | 数据集文件 | 样本数 | 标签分布 | 问题 | 状态 |
|------|-----------|--------|----------|------|------|
| **v1** | `router_no_reason_v1_4cls.json`（旧）| ~4813 | unsolvable ~62%, escalate ~19%, reflect ~10%, replan ~9% | unsolvable 严重主导；坍塌为 always_unsolvable（solve_rate 0.097）| ❌ 坍塌 |
| **v2** | `router_no_reason_v1_4cls.json`（全）| 13626 | unsolvable 60%, escalate 18%, reflect 11%, replan 11% | 同 v1，加剧；同样有 494 条标签冲突 | ❌ 坍塌 |
| **v3** | `router_no_reason_v1_4cls_deduped.json` | 7078 | escalate 30%, unsolvable 30%, reflect 20%, replan 20% | 去重（keep-first）+ cap unsolvable ≤30% | ⏳ 待训练 |

---

## 2. Benchmark 组合说明

Benchmark 数据由 `run_benchmark_exhaustive.py` 生成，对每道题跑全部 action 的 rollout，记录 `successful_actions`（哪些 action 能解题）。

| Benchmark 文件 | 题目数 | 有 SA 的题 | 分层方式 | 说明 |
|---------------|--------|------------|----------|------|
| `router_no_reason_v1_3cls_bench_random.json` | 400 | 328 | 随机抽样 | 3cls 主 eval |
| `router_no_reason_v1_3cls_bench_stratified.json` | 400 | 297 | 按来源数据集分层 | 覆盖更均匀 |
| `router_no_reason_v1_4cls_bench_random.json` | 400 | 318 | 随机抽样 | 4cls eval |
| `router_no_reason_v1_4cls_bench_stratified.json` | 400 | 319 | 按来源数据集分层 | — |
| `router_no_reason_v1_archB_4cls_bench_random.json` | 400 | 302 | 随机抽样 | archB（5-action）|

**Benchmark 标签的定义**：`cheapest_action(successful_actions)`，与训练数据标签逻辑一致。`solve_rate = 预测 action ∈ successful_actions 的比例`（比 dec_acc 高约 20pp，是正确指标）。

---

## 3. SFT 效果 vs 各种 baseline

### 动作 baseline（新指标 solve_rate，n=625 完整 rollout benchmark）

| 策略 | solve_rate | mean cost | 说明 |
|------|------------|-----------|------|
| always_reflect | 0.270 | 2.0 | 永远选最便宜动作，solve_rate 最低 |
| always_replan | 0.413 | 2.0 | — |
| random | 0.442 | ~5.7 | 随机选 action |
| always_escalate | 0.707 | 13.0 | 永远用大模型，solve_rate 高但 cost 贵 |
| **3cls v1 argmax** | **0.735** | 10.5 | 当前最好 SFT 模型 |
| oracle（完美路由）| 0.955 | 6.7 | 上界：总选 cheapest 有效 action |

**Oracle 说明**：oracle 不是一个模型，是数据本身决定的上界：

```
oracle solve_rate = 有 non-empty successful_actions 的题数
                   ─────────────────────────────────────
                   所有有 SA 记录的题数（含 empty）
```

benchmark 里约 4.5% 的题是真正无解的（reflect / replan / escalate 全部失败），即使路由完全正确也无法解决。cost=6.7 表示完美路由下的平均花费：约 43% 的题只有 escalate 能解（cost=13），其余用 reflect/replan 就够（cost=2）。

**结论**：3cls v1（0.735）与 oracle（0.955）之间有 **0.22 gap**，是路由质量还能提升的空间上界。always_escalate（0.707）几乎与 v1 持平，但 cost 高出 2.5×（13 vs 10.5）——SFT 的价值在于用更低 cost 达到相近甚至更高的 solve_rate。

---

## 4. SFT router vs GPT-based router

旧指标 dec_acc（n≈150，从原始 3cls SFT 数据的测试分割）：

| Router | dec_acc | mean cost | 说明 |
|--------|---------|-----------|------|
| GPT-nano argmax | 0.342 | 2.0 | CRC sweep 始终输出 reflect，无判别力 |
| GPT-mini argmax | 0.340 | 2.0 | 同上，tau=1.0 直到 alpha=0.5 才降 |
| Qwen3.5-4B base（无 SFT）| 0.358 | 2.0 | 同上 |
| **3cls v1 SFT argmax** | **0.522** | 10.6 | **+18pp** |

**为什么 GPT router 毫无用处**：GPT router 的 CRC sweep 在几乎所有 alpha 下 `chosen_dist = {reflect: n, replan: 0, escalate: 0}`，即 τ 从不收紧到能区分类别。模型对三个 action 的置信度分布太均匀（熵太高），无法产生有意义的路由。SFT 的核心价值就在于让模型学会区分 reflect / replan / escalate。

---

## 5. SFT 模型训练结果详情

| 模型 | solve_rate (random) | solve_rate (stratified) | 状态 |
|------|---------------------|-------------------------|------|
| 3cls v1 | **0.765** | 0.710 | ✅ 正常 |
| 3cls v2 | 0.283 | 0.269 | ❌ 全预测 reflect（361/400） |
| 3cls v3 | — | — | ⏳ 训练中 |
| 4cls v1 | 0.097 | 0.091 | ❌ 全预测 unsolvable（250/400，solve_correct=0） |
| 4cls v2 | 0.283 | 0.279 | ❌ 全预测 escalate（394/400） |
| 4cls v3 | — | — | ⏳ 待训练 |

**v3 训练配置**（3cls）：
- 数据：`router_no_reason_v1_3cls_deduped.json`（5114 条，去重后）
- 训练样本：4602（90%），验证：512（10%）
- 总步数：1728（3 epoch × 576 steps/epoch）
- 预估时长：~7-8 小时（2× V100 32GB）

---

## 6. Conformal Risk Control（CRC）

### 6.1 方法说明

路由策略为 `argmax_a [p(a|x) - λ · cost(a)]`，通过调节 λ 在 solve_rate 和 cost 之间权衡：
- λ=0：纯 argmax，只看置信度
- λ 增大：更倾向选便宜 action（reflect/replan），降低 cost

**CRC 的作用**：在 calib set 上找一个 λ̂，使 test set 上的期望 cost / fail rate 有概率保证。用 Hoeffding 不等式确定 slack ε，使得：

```
P(E_test[cost] ≤ C) ≥ 1 - δ     (Mode A)
P(E_test[fail] ≤ α) ≥ 1 - δ     (Mode B)
```

其中 δ=0.05，`ε = range × sqrt(log(1/δ) / 2n)`。

### 6.2 Calibration 数据

**旧 CRC**（`results/crc_v3_api_call.json`）：
- 来源：`conformal/data/calib.jsonl` / `test.jsonl`，从 3cls SFT 训练数据 80/10/10 分割
- n_calib=158，n_test=159
- 注意：与 SFT 训练集有重叠风险（LlamaFactory 的 val split 未隔离）

**新 benchmark CRC**（`results/crc_bench_3cls_v1.json`）：
- 来源：benchmark exhaustive rollout（625 道题全部跑了所有 action）
- n_calib=312，n_test=313，calib/test 随机 50/50 分割
- 与 SFT 训练集无重叠（benchmark 是独立构建的）
- ε_cost=0.762，ε_fail=0.069

### 6.3 Mode A — cost 预算约束（E[cost] ≤ C，以 95% 置信度）

基于新 benchmark CRC（n_calib=312，δ=0.05）：

| cost 预算 C | λ̂ | test solve_rate | test cost | 保证 |
|-------------|-----|-----------------|-----------|------|
| 5 | — | 0.562 | 4.32 | ✓ |
| 7 | — | 0.639 | 6.36 | ✓ |
| **8** | — | **0.665** | **7.31** | ✓ **sweet spot** |
| 9 | — | 0.690 | 8.22 | ✓ |
| 10 | — | 0.716 | 9.24 | ✓ |
| 12（≈argmax）| 0 | 0.735 | 10.47 | ✓ |

**C=8 的含义**：平均每题花费 ≤8 个单位（以 95% 置信度），代价是 solve_rate 从 0.735 降到 0.665（-7pp），换来 cost **节省 30%**（10.5→7.3）。

### 6.4 Mode B — fail rate 约束（E[fail] ≤ α，以 95% 置信度）

| α | 可行 | test solve_rate | test cost |
|---|------|-----------------|-----------|
| ≤0.60 | ✗ | — | — |
| **0.65** | ✓ | 0.460 | 2.0 |

α<0.65 不可行的原因：ε_fail=0.069，而 argmax 的 fail rate 本身是 0.265，target 必须 ≥ 0.265+ε ≈ 0.334 才有解。但 α=0.60 对应 target=0.531，理论上应该可行——实际不可行是因为 monotonicity correction（suffix max）使 fail 曲线在高 λ 处上升，挤压了可行区间。本质还是 **calib set 太小**（n=312）导致 ε 太大。

### 6.5 未来改进方向

| 优先级 | 方向 | 预期效果 |
|--------|------|----------|
| ① | **扩充 calib set 到 n≥500**（当前 n=312）| ε_fail 从 0.069 → <0.05，解锁 Mode B α=0.60 |
| ② | **用 v3 模型重跑 CRC** | 预期 argmax solve_rate 提升，Mode A 曲线上移 |
| ③ | **escalate imputation**：当 reflect=1 或 replan=1 时，impute O[escalate]=1 | 降低 fail rate 地板，Mode B 更容易可行 |
| ④ | **独立 calib set**（当前与 SFT 数据有潜在重叠）| 去掉分布偏差，保证数字可信 |

---

## 7. CRC 方法论版本（旧 v1/v2/v3，与 SFT 数据版本无关）

| 版本 | 方法 | 是否有保证 | 数据 |
|------|------|-----------|------|
| v1 | τ-prediction-set | ✓（但保证的是 label coverage，非 cost/fail，**错误目标**）| n=159 |
| v2 | λ-policy Pareto sweep | ✗ 纯探索，无保证 | n=159 |
| **v3** | λ-policy + Hoeffding CRC | ✓ Mode A E[cost]≤C / Mode B E[fail]≤α | n=159（旧）/ n=625（新 benchmark）|

当前以 **v3 + 新 benchmark CRC** 为主要结果。

---

## 文件索引

```
datasets/
  router_no_reason_v1_3cls.json              # 原始 3cls（5498条，含重复冲突）
  router_no_reason_v1_3cls_deduped.json      # 去重后 3cls（5114条）← v3 训练用
  router_no_reason_v1_4cls.json              # 原始 4cls（13626条）
  router_no_reason_v1_4cls_deduped.json      # 去重+cap后 4cls（7078条）← v3 待训练
  benchmarks/
    router_no_reason_v1_3cls_bench_random.json
    router_no_reason_v1_3cls_bench_stratified.json
    router_no_reason_v1_4cls_bench_*.json

sft_runs/
  router_arch_a_3cls.yaml / .sh              # 3cls v1 训练配置
  router_arch_a_3cls_v2.yaml / .sh           # 3cls v2（坍塌）
  router_arch_a_3cls_v3.yaml                 # 3cls v3（当前训练中）
  router_arch_a_4cls_v3.yaml                 # 4cls v3（待训练）
  outputs/
    router_arch_a_3cls/checkpoint-200-renamed  # 3cls v1 最佳 ← 当前使用
    router_arch_a_3cls_v2/checkpoint-1000       # 3cls v2（坍塌，勿用）
    router_arch_a_3cls_v3/                      # 训练中

conformal/results/
  bench_eval_3cls_v1/summary.json            # 3cls v1 benchmark eval
  bench_eval_3cls_v2/summary.json            # 3cls v2 benchmark eval（坍塌）
  bench_eval_4cls_v1/                        # 4cls v1（坍塌）
  bench_eval_4cls_v2/summary.json            # 4cls v2（坍塌）
  crc_bench_3cls_v1.json                     # 主 CRC 结果（新 benchmark，n=625）
  SUMMARY.md                                 # 数字速查表
```
