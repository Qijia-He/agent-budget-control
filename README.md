# Budget-Calibrated Recovery Routing for Coding Agents

**[Paper (arXiv)](https://arxiv.org/abs/TODO)**

When a cheap coding agent fails, should it reflect on the error, replan from scratch, or escalate to a stronger model? We train a supervised **recovery router** that picks among these three actions based on the problem and execution feedback, and wrap it with a **Conformal Risk Control (CRC)** layer that maps any target cost budget to a certified operating point — without retraining the router.

## Overview

After a cheap model fails on a coding task, the router observes `(problem statement, execution verdict, stderr)` and selects one recovery action:

| Action | Model | Behavior |
|---|---|---|
| `reflect` | cheap | repair code using the error trace |
| `replan` | cheap | discard attempt, solve from scratch |
| `escalate` | strong | strong model solves from scratch |

A Qwen3.5-4B router is fine-tuned on cheapest-successful-action labels from offline rollouts. At deployment, a scalar penalty λ is calibrated on a held-out split so that `argmax_a [s(a|x) - λ·c(a,x)]` satisfies a mean-cost budget *B* with a marginal finite-sample guarantee — no retraining needed as *B* changes.

**Main results (GPT-5.4-nano / GPT-5.4, n=360 held-out test examples):**

| Method | Solve rate | Mean cost (m$) |
|---|---|---|
| Always-reflect | 0.275 | 1.24 |
| Always-replan | 0.453 | 1.59 |
| Always-escalate | 0.686 | 7.22 |
| Binary cascade | 0.636 | 2.56 |
| **Ours — CRC (B = 2.56 m$)** | **0.717** | **2.56** |
| **Ours — CRC argmax (λ = 0)** | **0.817** | **5.51** |

## Repository Structure

```
agent-budget-control/
├── data_generation/          # Rollout collection pipeline
│   ├── agents/               # Code agent implementing reflect / replan / escalate
│   ├── benchmarks/           # Benchmark loaders and sandboxed execution environment
│   ├── core/                 # Runner, scorer, and pricing utilities
│   ├── scripts/              # Dataset building scripts (SFT formatting, soft labels, Gemini)
│   ├── launch/               # SLURM launch scripts for large-scale rollout collection
│   ├── llm_client.py         # Unified API client (OpenAI / Gemini)
│   └── requirements.txt
├── sft_runs/                 # Router training and eval configs (LLaMA-Factory format)
│   ├── qwen35_4b_full.yaml   # Primary model: Qwen3.5-4B full fine-tune (GPT setting)
│   ├── qwen3_4b_full.yaml    # Ablation: Qwen3-4B full FT
│   ├── qwen3_4b_lora.yaml    # Ablation: Qwen3-4B LoRA
│   ├── qwen3_8b_lora.yaml    # Ablation: Qwen3-8B LoRA
│   ├── gemini_qwen35_4b_full.yaml  # Cross-model: Gemini-2.5-Flash/Pro setting
│   ├── eval_qwen35_4b_full.sh      # Inference script for primary model
│   ├── eval_backbones.sh           # Inference script for ablation models
│   ├── eval_gemini.sh              # Inference script for Gemini setting
│   └── submit.sh                   # SLURM job launcher
├── conformal/                # CRC calibration and evaluation
│   ├── scripts/
│   │   ├── crc_on_holdout_usd.py       # Main CRC script (USD costs, 3-class)
│   │   ├── attach_usd_costs.py         # Attach API costs to router eval outputs
│   │   ├── compute_action_costs_usd.py # Compute mean per-action costs
│   │   ├── claude_predict.py           # Zero-shot Claude router baseline
│   │   ├── gemini_predict.py           # Zero-shot Gemini router baseline
│   │   └── nano_predict.py             # Zero-shot GPT-nano router baseline
│   └── data/
│       └── action_costs_usd.json       # Per-action mean cost estimates (millidollars)
├── data_analysis/
│   └── scripts/                        # Analysis and evaluation utilities
└── setup_env.sh
```

## Setup

```bash
bash setup_env.sh
pip install -r data_generation/requirements.txt
```

Router training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Install it separately and set the paths in the YAML configs to point to your installation and data directories.

## Pipeline

### 1. Collect recovery rollouts

For each problem, the cheap model produces an initial solution. If it fails, all three recovery actions are attempted and the outcomes are recorded.

```bash
bash data_generation/launch/run_apps_rollout.sh
bash data_generation/launch/run_apps_functional_rollout.sh
```

### 2. Build the SFT dataset

Label each example with the cheapest successful action and format for LLaMA-Factory:

```bash
python data_generation/scripts/build_sft_router_v1.py \
    --rollout_dir <rollout_dir> \
    --output_dir  <sft_data_dir>
```

For soft (weighted-copy) labels used in the ablation:

```bash
python data_generation/scripts/build_soft_label_dataset.py \
    --rollout_dir <rollout_dir> \
    --output_dir  <sft_data_dir>
```

### 3. Train the router

```bash
bash sft_runs/submit.sh qwen35_4b_full   # primary model
bash sft_runs/submit.sh qwen3_4b_lora    # ablation
```

The YAML configs follow LLaMA-Factory's format; adjust `data_dir`, `output_dir`, and `model_name_or_path` for your environment.

### 4. Run router inference on the holdout

```bash
bash sft_runs/eval_qwen35_4b_full.sh
```

This writes per-example action log-probabilities to a results directory.

### 5. Attach API costs and run CRC

```bash
python conformal/scripts/attach_usd_costs.py \
    --eval_dir <eval_results_dir>

python conformal/scripts/crc_on_holdout_usd.py \
    --calib <calib_eval.json> \
    --test  <test_eval.json> \
    --budget 2.56
```

`crc_on_holdout_usd.py` computes the breakpoint grid Λ_K from the calibration split, selects the smallest feasible λ̂ for the target budget, and reports solve rate and mean cost on the held-out test split.

### Zero-shot router baselines

```bash
python conformal/scripts/claude_predict.py  --eval_json <eval.json>
python conformal/scripts/gemini_predict.py  --eval_json <eval.json>
python conformal/scripts/nano_predict.py    --eval_json <eval.json>
```

## Citation

```bibtex
@article{he2025budget,
  title   = {Budget-Calibrated Recovery Routing for Coding Agents},
  author  = {He, Qijia and Cheng, Jiayi and Le, Chenqian and Wang, Rui and
             Liu, Xunmei and Chen, Yixian and Mei, Jie and Wang, Zhihao and
             Chen, Xupeng and Chen, Yuhuan and Wang, Tao},
  journal = {arXiv preprint},
  year    = {2025}
}
```
