# SFT data — single-step cascade router (paper #1, v1 no-reasoning ablation)

This folder contains the SFT data used to train the cost-aware cascade router.
Two architectures (A / B) × two label-set sizes (drop-unsolvable vs keep-unsolvable)
= 4 LlamaFactory-format JSON files.

## TL;DR

| Variant | File | Examples | Input | Labels |
|---|---|---|---|---|
| **Arch A 3cls** | [`router_no_reason_v1_3cls.json`](router_no_reason_v1_3cls.json) | 5498 | problem + verdict + stderr | reflect / replan / escalate |
| **Arch A 4cls** | [`router_no_reason_v1_4cls.json`](router_no_reason_v1_4cls.json) | 13626 | problem + verdict + stderr | + unsolvable |
| **Arch B 4cls** | [`router_no_reason_v1_archB_4cls.json`](router_no_reason_v1_archB_4cls.json) | 19172 | problem only | proceed / reflect / replan / escalate |
| **Arch B 5cls** | [`router_no_reason_v1_archB_5cls.json`](router_no_reason_v1_archB_5cls.json) | 27300 | problem only | + unsolvable |

Previous smaller versions of these files are archived in [`old/`](old/).

## What the router decides

The cascade has four actions, ordered by cost in **nano-units** (gpt-5.4-nano = 1):

| action | cost | meaning |
|---|---|---|
| `proceed` | 1 | accept small model's initial attempt |
| `reflect` | ~2 | small model fixes its own code given failure trace |
| `replan` | ~2 | small model discards attempt and re-plans from scratch |
| `escalate` | ~12 | strong model (gpt-5.4) re-solves from scratch |

Plus a 5th "label" (Arch B only, or Arch A 4cls): `unsolvable` — no cascade action will succeed; don't spend budget.

## Labeling rule

**Cheapest action whose verdict == pass.** For each rollout record:

1. Look at `summary.successful_actions` — list of actions that produced a passing run.
2. Pick the cheapest by `COST = {proceed: 1, reflect: 2, replan: 2, escalate: 13}`.
3. Cost-ties (reflect vs replan, both cost 2) broken by `PRIO = [proceed > reflect > replan > escalate]`.
4. If `successful_actions == []` → `oracle_unsolvable`. Handled per-variant (dropped, or kept as label `unsolvable`).

The label is therefore **the action a perfect oracle router would pick** to minimize cost.

Code: `cheapest_action()` in both build scripts.

## Architecture A vs B (key difference)

**Arch A** = post-proceed router. Cascade pipeline:

```
problem → proceed → verdict
  if pass → done                      (cost = 1)
  if fail → ROUTER picks {reflect, replan, escalate, unsolvable}
                  → run that         (cost = 1 + selected_action_cost)
```

The router only fires on **proceed-failure**, so the SFT input includes the proceed failure signal:
```
Problem:
<problem prompt, capped 6000 chars>

Initial attempt verdict: <fail|timeout|compile_error>
Initial attempt stderr:
<stderr, capped 800 / 400 chars>
```

`verdict == pass` rollout records are filtered out — they'd never invoke the router at deployment. That's why Arch A files don't contain a `proceed` label.

**Arch B** = pre-proceed router. Cascade pipeline:

```
problem → ROUTER picks {proceed, reflect, replan, escalate, unsolvable}
  → run that action directly         (cost = selected_action_cost only)
```

The router fires once, before any LLM call, with only the problem as input:
```
Problem:
<problem prompt, capped 6000 chars>
```

`proceed` is a meaningful label here — "this looks easy, just run nano". `unsolvable` is a "don't bother" call.

### Cost comparison (perfect router)

Using current data distribution: proceed-pass 39%, reflect 7%, replan 5%, escalate 8%, unsolv 41%:

| arch | avg cost / problem | notes |
|---|---|---|
| naive cascade (always full sequence) | ~17 | proceed + reflect + replan + escalate |
| Arch A (perfect post-proceed router) | **~7.1** | always pays proceed=1; saves on recovery |
| Arch B (perfect, retries unsolvable) | **~6.5** | saves wasted proceed on direct-escalates |
| Arch B (perfect, skips unsolvable) | **~1.6** | gives up on 41% of problems |

Arch A is the safer headline; Arch B is the bigger upside if the router can predict difficulty from problem text alone. We train both as ablation.

## How to (re)build

### Source data

```
outputs/rollouts/v55/
  bcb.jsonl              ← BigCodeBench (1140 problems)
  lcb.jsonl              ← LiveCodeBench release_v6 (1055 problems)
  taco_medhard.jsonl     ← TACO MEDIUM_HARD (2176 problems)
  taco_veryhard.jsonl    ← TACO VERY_HARD (2300 problems)
  taco_hard.jsonl        ← TACO HARD (in progress)
  taco_medium.jsonl      ← TACO MEDIUM (queued)
```
Each JSONL line is a short-circuit rollout record (proceed → reflect+replan → escalate) with `summary.successful_actions` + per-call verdicts.

### Build scripts

```bash
cd /cloudide/workspace/conformal_react/code
source ../.venv/bin/activate

# Arch A (post-proceed router, drops verdict=pass)
python scripts/build_sft_router_v1.py            # incremental — picks up new records
python scripts/build_sft_router_v1.py --rebuild  # full rebuild from scratch

# Arch B (pre-proceed router, problem-only input)
python scripts/build_sft_router_v1_archB.py
python scripts/build_sft_router_v1_archB.py --rebuild
```

Both scripts are **resumable**: they track processed `(dataset_tag, problem_id)` pairs in `.build_state.json` / `.build_state_archB.json` and only build new examples on re-run. Each run takes ~1 second.

### Phase A still running

`outputs/rollouts/v55/` is the output of Phase A — the large short-circuit rollout (tmux session `confo-phase4`, see [`../../run_phase4_all.sh`](../../run_phase4_all.sh)). taco_hard + taco_medium will add ~6k more rollout records. Re-run the build scripts after each tier completes to absorb them.

Check Phase A progress:
```bash
wc -l outputs/rollouts/v55/*.jsonl
tail outputs/rollouts/v55/logs/master.log
ls outputs/rollouts/v55/logs/PHASE4.DONE  # marker on completion
```

## LlamaFactory integration

The combined `dataset_info.snippet.json` registers all 4 variants:

```bash
# copy data + paste registration
cp outputs/sft/router_no_reason_v1_*.json /path/to/LLaMA-Factory/data/
# then merge dataset_info.snippet.json into /path/to/LLaMA-Factory/data/dataset_info.json
```

In your LlamaFactory training YAML:
```yaml
dataset: router_no_reason_v1_3cls        # or _4cls / _archB_4cls / _archB_5cls
template: qwen3
cutoff_len: 4096                          # p99 input ≈ 4.5k chars ≈ 1.2k tokens, plus instruction
```

## File index

| File | Purpose |
|---|---|
| `router_no_reason_v1_3cls.json` | Arch A, 3 classes (drop unsolvable) |
| `router_no_reason_v1_4cls.json` | Arch A, 4 classes (with unsolvable) |
| `router_no_reason_v1_archB_4cls.json` | Arch B, 4 classes (drop unsolvable) |
| `router_no_reason_v1_archB_5cls.json` | Arch B, 5 classes (with unsolvable) |
| `old/` | Previous smaller versions of all 4 files (1583 / 4813 / 4712 / 7956 examples) |
| `dataset_info.snippet.json` | LlamaFactory `dataset_info.json` snippet for all 4 |
| `dataset_info.archB.snippet.json` | Subset — Arch B only (kept for cleanliness) |
| `.build_state.json` | Arch A resume state — do not edit |
| `.build_state_archB.json` | Arch B resume state — do not edit |
| `README.md` | this file |

## v2+ ablation knobs (future)

Things we deliberately did NOT add in v1:

- **Reasoning output** — outputs are 1-word labels. v2 will add reasoning (e.g., diagnose's `failure_reason` then action) and we'll measure if it helps.
- **Proceed code in input (Arch A)** — without reasoning, the model can't intelligently read 2k chars of code. v3 will pair "with code" + "with reasoning".
- **Soft labels for cost-tied wins** — currently hard-labels reflect when reflect == replan == 2. Soft labels (0.5/0.5) would be a marginal upgrade.
- **Phase B datasets** — DS-1000 / SciCode / HumanEvalFix / DebugBench probes are pending. If any show useful cascade signal, they'll be added to the rollout and absorbed via re-running the build scripts.
