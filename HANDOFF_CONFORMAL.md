# Handoff — adding the conformal (CRC) layer to the trained router

> **You are**: another Claude Code instance on a different platform, with access to a Qwen3.5 router checkpoint trained on ~1k SFT examples.
> **Your task**: implement the conformal risk control (CRC) layer that sits *on top of* the trained router and turns it into a deployment-time, budget-controllable system.
>
> This doc is fully self-contained — you don't need to read the full `auto_planning_agent_conformal.md` plan, but everything important from it is summarised here. Pointers to the canonical code/data files are at the bottom.

---

## 0. One-paragraph project pitch

We are training a **cost-aware cascade router** for code generation. A cheap model (gpt-5.4-nano, cost=1) tries the problem first. If it fails, the router picks the cheapest *recovery action* — `reflect` (let nano fix its own code given the stderr; cost ~2), `replan` (let nano restart with a new plan; cost ~2), `escalate` (let gpt-5.4 re-solve from scratch; cost ~12), or `unsolvable` (no action will help; give up). The router's input is the problem prompt + verdict + stderr from the proceed attempt. The Qwen3.5 router checkpoint you have was SFT'd with cheapest-passing-action labels.

**Conformal layer (your job)** wraps the router with a calibration step so the operator can specify a **target risk α** at deployment time and the system guarantees the actual selection-error rate stays ≤ α (in expectation, with finite-sample bounds). This turns the router into a *budget-responsive* system instead of a fixed policy: at tight budget the operator raises α → CRC shifts the router toward cheaper actions with a guaranteed bound on quality loss.

---

## 1. What the router does (input/output)

The router is a Qwen3.5 model fine-tuned in the LlamaFactory Alpaca format. **Input** = problem + proceed-attempt result; **output** = one of 3-5 action labels.

### 1.1 Concrete instance (Arch A, 3-class) — this is most likely what was trained

**System instruction** (fixed, identical for every example):
```
You are a cost-aware coding router. A small fast model just attempted the
coding problem below and FAILED. Based on the problem and the failure trace,
choose the cheapest recovery action that will solve it.

Recovery actions (in increasing cost):
  reflect  — small model fixes its own code given the failure trace (cost ~1.2)
  replan   — small model discards the attempt and re-plans from scratch (cost ~1.5)
  escalate — strong model solves from scratch (cost ~12)

Reply with exactly one word: reflect, replan, or escalate. Do not explain.
```

**User input** (varies per example):
```
Problem:
<problem prompt, capped 6000 chars>

Initial attempt verdict: <fail | timeout | compile_error>
Initial attempt stderr:
<stderr, capped 800 chars>
```

**Assistant output** (the label to learn): `reflect`, `replan`, or `escalate` — single word.

If the trained checkpoint was the 4-class variant, add `unsolvable` to the action set; if Arch B was used, the input is problem only (no verdict/stderr) and labels include `proceed` (cost 1).

**The router was trained without reasoning output**, just the label.

### 1.2 Cost ratios

In *nano-units* (gpt-5.4-nano API cost = 1):
```python
COST = {"proceed": 1.0, "reflect": 2.0, "replan": 2.0, "escalate": 13.0}
```
Tie-break order for cheapest-label: `[proceed > reflect > replan > escalate]`.

---

## 2. What CRC adds, and why it matters

### 2.1 Motivation (the deployment knob)

Without CRC, the router is a static `argmax` classifier. It picks one action per problem. The cost-quality tradeoff is whatever the SFT happened to produce — there is no way to tune it at deployment.

CRC turns the router into a **budget-responsive** policy: the operator specifies an escalation-budget (or failure-budget) constraint at deployment time, and CRC calibrates a single scalar parameter so the constraint holds on exchangeable test data with finite-sample guarantee.

This is the **paper #1 headline differentiator vs RouteLLM / ToolOrchestra**: they give one Pareto point per training run; we give the whole Pareto frontier with calibration guarantees, from one trained router.

### 2.2 The CRC policy form (this is the actual paper design)

CRC does **not** use the standard "prediction set with coverage 1−α" framing. The paper uses a **cost-regularized argmax with a single calibrated scalar λ**:

```
policy_λ(x) = argmax_a [ p_a(x) − λ · cost(a) ]

  λ = 0    : cost-blind (raw argmax → highest accuracy, highest cost)
  λ → ∞   : cheapest action regardless of confidence
```

Where:
- `p_a(x)` ∈ [0, 1] = router's **temperature-scaled** softmax probability for action `a` (see §3.3 — this is critical, not optional).
- `cost(a)` = cost of action `a` in nano-units (proceed=1, reflect=1.2, replan=1.5, escalate=8 — paper's numbering; see §1.2 for the canonical table).
- λ ≥ 0 = the conservativeness knob CRC will calibrate.

### 2.3 Two operating modes (both are CRC-calibrated)

**Primary — escalation-budget mode (cost-constrained):**
```
maximize  Pr[success]      s.t.  Pr[escalate] ≤ C
```
Sweep λ on the calibration split. Pick the **smallest λ** such that `Pr[escalate at policy_λ] ≤ C`. That's the most accuracy-favourable policy that still respects the budget. CRC's finite-sample guarantee transfers directly: the same `Pr[escalate]` bound holds on exchangeable test data, up to standard CRC slack (~ O(1/√n_cal)).

**Dual — failure-risk-budget mode (quality-constrained, free byproduct):**
```
minimize  E[cost]          s.t.  Pr[fail] ≤ α
```
Same sweep, different stopping rule. Pick the **largest λ** such that `Pr[fail at policy_λ] ≤ α`. We report both operating points — production teams pick whichever constraint is binding.

> **Monotonicity caveat — important.** Primary mode (Pr[escalate]) IS strictly monotone in λ: as λ ↑, escalate gets penalized more, so escalate rate must decrease. The monotonicity is exact (not just empirical), so the breakpoint sweep cleanly finds the optimal λ. **Dual mode (Pr[fail]) is NOT theoretically monotone**: as λ ↑ the policy substitutes cheaper actions, which usually but not always increases fail rate. On the n=30 ablations there are cases where replan succeeds and escalate fails on the same problem, so swinging λ from "escalate-favorable" to "replan-favorable" can *decrease* the fail rate. In practice the dual sweep produces a near-monotone curve with small dips, and you should pick "largest λ such that Pr[fail] ≤ α holds at every λ' ≥ λ in the sweep" rather than the literal largest crossing. Report this caveat explicitly in the paper — primary mode (escalate-budget) is the cleaner theoretical story; dual is empirical.

### 2.4 Why a single λ instead of per-action thresholds

The n=30 ablation (see `code/outputs/bigcodebench/code_bigcodebench_gradient.md`) shows the actions are **not** monotone in difficulty: replan can succeed where escalate fails. Per-action thresholds assume a quality ordering that doesn't exist. The cost-regularized argmax breaks this — λ trades off `Δp` against `Δcost` between any two actions, no ordering needed.

### 2.5 Frontier shape (honest framing)

Because `policy_λ` is an argmax, it's a **step function** in λ — the chosen action flips at a finite set of breakpoints as λ sweeps from 0 to ∞. The achievable cost-accuracy frontier is therefore a **discrete set of operating points**, not a smooth curve. In principle there are up to `4 · N_cal` breakpoints; in practice with label smoothing + temperature scaling the realized count is dozens to a few hundred. **Report the breakpoints explicitly** (don't plot a smoothed curve that suggests continuity the policy class doesn't deliver). The CRC guarantee transfers cleanly to "the chosen discrete operating point".

---

## 3. The concrete CRC pipeline (what to implement)

There are **two calibration steps**: temperature scaling first (Step 3.3), then CRC λ (Step 3.4). Both consume the same calibration split — no extra budget needed.

### 3.1 End-to-end data flow

```
trained_router_ckpt + calibration split (problems with ground-truth cheapest labels)
        │
        ├─ Step 3.2: forward pass to extract per-example
        │            candidate_logits ∈ ℝ^{N_cal × K}   (K = 3, 4, or 5 actions)
        │            ground_truth_idx ∈ {0..K-1}
        │
        ├─ Step 3.3: fit temperature τ on (logits, labels) via LBFGS minimising NLL.
        │            Save τ alongside checkpoint.
        │
        ├─ Step 3.4: with p_a(x) = softmax(logits / τ), sweep λ from 0 to large.
        │            For each λ on calibration set, compute Pr[escalate at policy_λ]
        │            and Pr[fail at policy_λ]. Pick the constraint-satisfying λ.
        │
        └─ Step 3.5: at deployment, score new x → softmax(logits/τ) → policy_λ(x).
```

### 3.2 Score extraction (single-token logit readout)

The router was trained with the answer being a single letter token (`A`/`B`/`C`/`D` for proceed/reflect/replan/escalate). To get calibratable probabilities:

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
model = AutoModelForCausalLM.from_pretrained(
    CKPT_PATH, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

# Map each action-letter to its single token id. Verify single-token-ness once:
candidate_strs = [" A", " B", " C", " D"]   # leading space matters in BPE
ABCD_token_ids = [
    tokenizer.encode(s, add_special_tokens=False)[-1]
    for s in candidate_strs
]
# assert each encode result has length 1; otherwise see §9 risk notes.

def candidate_logits(prompt: str) -> torch.Tensor:
    """Return last-position logits restricted to {A,B,C,D} — shape [4]."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        full_logits = model(**inputs).logits[0, -1, :]   # vocab-sized last-position
    return full_logits[ABCD_token_ids].float()
```

**Important** — the SFT data may have been built with `reflect / replan / escalate` as the literal output strings, not letter tokens. **Check the training data format first.** If it uses words, you'll need to score the *first token of each word* (e.g. `" reflect"[0]`), and verify those first tokens are unique across actions. If they collide, switch to multi-token scoring (sum log-probs over the action's token sequence) — less clean but still works.

### 3.3 Temperature scaling (mandatory prerequisite to CRC)

Hard-label SFT (with K-1 negative-log-likelihood) produces near-degenerate `p_argmax ≈ 0.99` on the trained set. With probabilities that peaked, `λ · cost(a)` can't compete: the policy is stuck in raw-argmax mode for all reasonable λ, and the achievable frontier collapses to 2-3 points.

Fit a single scalar temperature `τ` on the calibration split to soften:

```python
# offline, one-shot, runs in seconds
candidate_logits_cal = torch.stack([
    candidate_logits(prompt_i) for prompt_i in cal_set
])                                                            # [N_cal, K]
y_cal = torch.tensor([action_idx(i) for i in cal_set])        # [N_cal]

log_tau = torch.nn.Parameter(torch.zeros(1))                  # log-space, init τ=1
opt = torch.optim.LBFGS([log_tau], lr=0.01, max_iter=50)

def closure():
    opt.zero_grad()
    tau = log_tau.exp()
    loss = F.cross_entropy(candidate_logits_cal / tau, y_cal)
    loss.backward()
    return loss

opt.step(closure)
tau_hat = log_tau.detach().exp().item()
# Persist tau_hat next to the checkpoint.
```

**Typical post-SFT τ lands in 1.5–3.0.** This pulls `p_argmax` from ~0.99 to ~0.7–0.8 and gives the other candidates ~0.05–0.15 each — enough dynamic range for λ to do its job.

> **Diagnostic: if τ̂ > 5, the SFT overfit and temperature scaling alone won't fully fix it.** The Pareto frontier under CRC will collapse to a small handful of operating points. Mitigations, in order of cost:
> 1. **Re-train SFT with label smoothing** (e.g. `label_smoothing=0.1` in the LlamaFactory config) and/or stronger dropout / fewer epochs. This is the right fix.
> 2. **Coarsen the label set** — collapse reflect+replan into a single "cheap-recovery" class. The trained router has less to learn, less to overfit. Loses the reflect-vs-replan distinction but gains calibration.
> 3. **Accept the smaller frontier** — fewer operating points, narrower α/C range. Document the limitation and ship.

For SFT examples with **soft targets** (Step 1 in the paper-plan emits soft targets for cost-tied cheapest actions), replace the cross-entropy closure with the soft-target NLL:
```python
log_p = F.log_softmax(candidate_logits_cal / tau, dim=-1)
loss = -(q_target_cal * log_p).sum(dim=-1).mean()
```
If you're not sure whether soft targets were used, the integer-target form above is fine for the hard-label majority.

Temperature scaling is independent of CRC — it only reshapes `p_a(x)`. The downstream CRC finite-sample guarantee operates on the scaled probabilities unchanged; no re-derivation or extra calibration budget needed.

### 3.4 CRC λ calibration

With temperature-scaled `p_a(x)` per calibration example and a cost vector, find the constraint-satisfying λ.

```python
import numpy as np

# cal_probs[i, a] = p_a(x_i) after temperature scaling      # [N_cal, K]
# cal_outcomes[i, a] ∈ {0, 1} = 1 if action a passes on problem i  (from rollouts)
# cost_vec[a] = cost of action a                             # [K]

ESCALATE_IDX = K - 1   # by convention, last action is escalate

def policy_lambda(probs, lam):
    """Vectorized: returns chosen action per row."""
    return np.argmax(probs - lam * cost_vec, axis=-1)        # [N_cal]

def metrics(lam):
    chosen = policy_lambda(cal_probs, lam)
    # selected action's outcome per problem
    success = cal_outcomes[np.arange(len(chosen)), chosen]
    pr_escalate = (chosen == ESCALATE_IDX).mean()
    pr_fail = 1 - success.mean()
    avg_cost = cost_vec[chosen].mean()
    return pr_escalate, pr_fail, avg_cost

# Build the set of candidate λ values to sweep. At argmax, the policy only
# flips at breakpoints where two actions tie:
#   p_a − λ·cost(a)  =  p_b − λ·cost(b)
# ⇒  λ_break(a,b; x) = (p_a(x) − p_b(x)) / (cost(a) − cost(b))   for cost(a) ≠ cost(b)
# Enumerate all breakpoints across calibration set, sort, sweep midpoints between
# adjacent breakpoints. That gives the exact discrete frontier.
breakpoints = []
for i in range(N_cal):
    p = cal_probs[i]
    for a in range(K):
        for b in range(K):
            if a != b and cost_vec[a] != cost_vec[b]:
                lam = (p[a] - p[b]) / (cost_vec[b] - cost_vec[a])
                if lam > 0:
                    breakpoints.append(lam)
lams = sorted(set(breakpoints))
# Add midpoints to land cleanly inside each interval
sweep_lams = [0.0] + [
    (lams[i] + lams[i + 1]) / 2 for i in range(len(lams) - 1)
] + [lams[-1] * 1.5 if lams else 1e6]

# --- Primary objective: smallest λ with Pr[escalate] ≤ C ---
C = 0.20   # operator-chosen budget; sweep this externally for the figure
lam_star = None
for lam in sweep_lams:
    pr_esc, pr_fail, avg_cost = metrics(lam)
    if pr_esc <= C:
        lam_star = lam
        break

# --- Dual objective: largest λ with Pr[fail] ≤ α ---
ALPHA = 0.10
lam_star_dual = None
for lam in reversed(sweep_lams):
    pr_esc, pr_fail, avg_cost = metrics(lam)
    if pr_fail <= ALPHA:
        lam_star_dual = lam
        break
```

The sweep is O(N_cal · K² · log(N_cal · K²)) — for N_cal ≈ 500 and K=4, that's a few thousand candidate λ values; runs in seconds on a laptop.

**CRC finite-sample correction** — pick one of three; document the choice:

The empirical Pr[escalate] / Pr[fail] on the calibration set is biased: λ̂ is chosen to satisfy the empirical constraint, so by definition the empirical estimate at λ̂ sits on the boundary and the true value can exceed it. Three valid corrections:

> **(a) Data splitting.** Split the calibration set in two halves: pick λ̂ on the first half, *verify* `Pr[escalate at policy_λ̂] ≤ C` on the second half (no further selection). Cleanest theoretical guarantee — no slack needed — but halves the effective calibration size.
>
> **(b) Hoeffding slack adjustment** (recommended for n ≥ 500). Instead of `\hat{Pr}_cal ≤ C`, require `\hat{Pr}_cal ≤ C − ε(n, δ)` where `ε(n, δ) = √(log(2/δ) / (2n))`. For n=500, δ=0.05, ε ≈ 0.07. Eats some of the budget but is concentration-based and tight asymptotically. Guarantee: `Pr_test[escalate] ≤ C` w.p. ≥ 1−δ.
>
> **(c) Union bound over breakpoints.** Because λ only takes O(K²·n_cal) distinct values via breakpoints, use δ' = δ / N_breakpoints. Adds `log(N_breakpoints)` ≈ 5-10 to the slack vs (b) but lets you certify the *entire* λ-sweep simultaneously. Useful if you want the figure to validly report multiple operating points.

Method (b) is the right default for our scale. With n_cal ~500-1000, ε ≈ 5-7%, and we'd ship "C = 0.20 with guarantee" by selecting on `\hat{Pr}_cal ≤ 0.13-0.15`.

### 3.5 Deployment

```python
def route(x: str, lam_star: float, tau_hat: float) -> str:
    logits = candidate_logits(x)
    probs = F.softmax(logits / tau_hat, dim=-1).cpu().numpy()
    action_idx = int(np.argmax(probs - lam_star * cost_vec))
    return ACTIONS[action_idx]
```

That's the entire CRC-routed deployment path.

### 3.6 Calibration / test data split

The SFT data is split 90/10 train/eval by default in LlamaFactory. **Don't use the eval split for both calibration and test** — it'll inflate the apparent CRC guarantee.

Pragmatic choice: split the **existing SFT data 80/10/10** → train / calibration / test. The calibration set needs to be i.i.d. with deployment data (same problem-source distribution); if the production target is APPS / cross-dataset, calibrate on a held-out APPS split too.

For each example we need `cal_outcomes[i, a] ∈ {0,1}` (success per action) — that comes from the original rollout records in `code/outputs/rollouts/v55/*.jsonl` field `summary.successful_actions`. Just look up the rollout corresponding to each calibration problem_id.

---

## 4. Existing code in this repo — what to use and what to ignore

The `core/` folder has several CRC-related files from earlier iterations of the project. **Most of them implement the wrong CRC framing** (binary trust-or-escalate, prediction-set coverage). Don't reuse them blindly. Specifically:

| File | Verdict | Reason |
|---|---|---|
| `core/calibration.py` | **Do not reuse** | Implements `s ≤ τ` thresholding for binary 2-class case. The λ-CRC in §2 is a different primitive — single-scalar over cost-regularized argmax, not a threshold on a nonconformity score. |
| `core/conformal_layer.py` | **Do not reuse** | Designed for sequential agentic step routing with a 2-decision (proceed / reflect / escalate) split. Paper #1 is single-shot multi-action routing. |
| `core/nonconformity.py` | **Do not reuse** | Step-level placeholder; wrong abstraction for the multi-class router. |
| `core/pricing.py` | ✓ reusable | Cost table. But verify cost numbers — paper uses {1, 1.2, 1.5, 8}; existing SFT-build code uses {1, 2, 2, 13}. See §1.2 + §9 risk note. |
| `core/short_circuit_runner.py` | ✓ reference only | Shows how cheapest-action labels are produced from real rollouts. You'll need to map calibration-set problem IDs back to rollout records here to get the `cal_outcomes` matrix in §3.4. |

**New code you should write** lives in fresh files. Suggested layout:

```
code/scripts/
  router_extract_logits.py   # Step 3.2 — run trained ckpt on cal/test splits, dump logits
  fit_temperature.py         # Step 3.3 — LBFGS τ on calibration logits
  calibrate_crc.py           # Step 3.4 — sweep λ, pick constraint-satisfying value
  eval_crc.py                # Step 3.5 — replay test set with (τ, λ), compute metrics
                             # + sweep C ∈ {0.10, ..., 0.30} for the Pareto figure

code/core/
  crc_router.py              # the single deployment-time route() function (3.5)
```

---

## 5. SFT data — what the router was trained on

Folder: `code/outputs/sft/`. Detailed schema: `code/outputs/sft/README.md`. Summary:

| Variant | File | Examples | Input shape | Labels |
|---|---|---|---|---|
| Arch A 3cls | `router_no_reason_v1_3cls.json` | ~1.6k → ~2.3k (grew with Phase A) | problem + verdict + stderr | reflect / replan / escalate |
| Arch A 4cls | `router_no_reason_v1_4cls.json` | ~4.8k → ~7k | + unsolvable |
| Arch B 4cls | `router_no_reason_v1_archB_4cls.json` | ~4.7k → ~7k | problem only | proceed / reflect / replan / escalate |
| Arch B 5cls | `router_no_reason_v1_archB_5cls.json` | ~8k → ~11k | + unsolvable |

The ~1k checkpoint you have was almost certainly trained on the 3cls variant before Phase A finished accumulating data. Sizes will keep growing (APPS rollout is in flight, adds ~3000 more failure cases).

Labeling rule: **cheapest action whose verdict == pass**, ties broken by priority. Code in `code/scripts/build_sft_router_v1.py` function `cheapest_action()`.

---

## 6. The big picture (paper #1 narrative)

The story we want to tell:

1. **Cascade routing for code is real**: empirically, on hard datasets like TACO MEDIUM_HARD, escalate has a 17% unique niche that cheaper actions can't reach (B3-only); on saturated datasets like HumanEval / MBPP it has zero niche. So a *static* policy (always-cascade, always-escalate, …) is leaving a systematic cost-quality gap.

2. **A learned router can capture this**: we train Qwen3.5 on cheapest-passing-action labels from 11k+ real cascade rollouts. SFT-only, no RL. The router learns to map `(problem, proceed-failure-trace)` → cheapest recovery action.

3. **CRC turns the router into a budget-responsive system** *(your part)*: the conformal layer lets operators slide α at deployment time, giving a calibrated Pareto curve instead of a fixed policy. This is the differentiation against RouteLLM (Ong et al. 2024) and ToolOrchestra (NVIDIA 2025) — both produce one Pareto point per training run.

The combination is the paper. Without CRC, point 3 collapses and we'd be a small-margin improvement on RouteLLM. **The conformal layer is load-bearing for the contribution**.

---

## 7. Concrete deliverables

### 7.0 **Critical-path ordering — read this before writing code**

**Do not jump straight to CRC.** The whole story collapses if any of the precondition steps fail. Validate in this order:

1. **Validate the trained router beats trivial baselines** (highest priority — paper's life or death).
   Run the trained router (raw argmax, no CRC, no temperature scaling yet) on a held-out test set. Compute:
   - per-class confusion matrix
   - top-1 accuracy (chosen action passes? — needs the outcomes matrix from §9)
   - average cost vs always-escalate (cost ≈ 12) and heuristic 2-tier (proceed-then-escalate, cost ≈ 1 + 12·p(proceed-fail))

   **Pass criterion**: trained router (λ=0 raw argmax) strictly dominates heuristic 2-tier on the cost-accuracy plane. If not, CRC can't save it — CRC slides along a Pareto frontier, it cannot push the frontier toward better cost-accuracy. Action: scale SFT (we have 11k+ examples after Phase A, ~20k after APPS rollout completes), retrain, re-validate.

2. **Acquire an outcomes-complete calibration set** (~500-1000 problems, see §9 "Outcomes matrix").
   The existing n=30 independent A/B/C/D sweeps give only ~380 problems total → CRC slack ~7% (Hoeffding at δ=0.05), which eats most of any reasonable C / α budget. Run an additional independent A/B/C/D sweep on **500-1000 random Phase A problems** before doing CRC calibration — cost ~$15-25, wall ~6-8h. Skip this and the CRC guarantee is meaningless.

3. **Temperature scaling** (Step 3.3). Check τ̂ lands in 1.5-3.0; if τ̂ > 5 see §3.3 + §9.

4. **CRC λ calibration** (Step 3.4).

5. **Figure + validation table** (deliverables 5 + 6 below).

**Only steps 1, 3, 4, 5 are pure compute — step 2 requires running additional API rollouts.** Coordinate with the upstream rollout team if needed; the script is `code/scripts/collect_rollouts_opt.py` but it needs to be configured for `--no-short-circuit` mode (currently does not exist — needs a small patch to run all four actions per problem independently).

### 7.1 Code/file deliverables

1. **`scripts/router_extract_logits.py`** — dump candidate logits for each problem in the cal+test splits. Output jsonl: `{problem_id, candidate_logits: [4 floats], true_label}`. Also record per-example success-per-action `[1,1,0,1]` etc. by joining to the rollout file.

2. **`scripts/fit_temperature.py`** — fit `τ` via LBFGS on calibration logits. Save `tau_hat.json`.

3. **`scripts/calibrate_crc.py`** — given `tau_hat` and an escalation-budget `C` (or failure-budget `α`), compute λ* via the breakpoint sweep in §3.4. Save `crc_lambda.json` containing `{lam_star, tau_hat, C, alpha, n_calib, finite_sample_slack}`.

4. **`scripts/eval_crc.py`** — replay the test split with `(τ̂, λ̂)`. For the **Pareto figure**, sweep `C ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` (and symmetrically the dual `α` sweep), get one λ* per C, evaluate on test, record `(avg_cost, success_rate, pr_escalate_actual)`. Each row of the table is one C-value operating point.

5. **Main figure (cost-accuracy plane)**. x = avg cost/problem (nano-units), y = solve rate. Plot:
   - always-proceed (cost=1, low quality) — anchor left
   - always-escalate (cost=12) — anchor upper-right
   - naive cascade run-all-actions (cost~17) — strictly dominated reference
   - trained router with raw argmax (λ=0) — single point
   - **CRC router with C sweep — a sequence of discrete operating points (step-function frontier)**
   - **CRC router with α sweep — overlay**, same color but different markers (the dual mode)

6. **Validation table** — for each operating point, report the *actual* test-set `Pr[escalate]` (escalation-budget mode) or `Pr[fail]` (failure-budget mode) vs the constraint. The CRC guarantee is that empirical exceedance probability ≤ δ (the finite-sample slack you used in §3.4).

If any of the score-extraction details in §3.2 don't match how the router was actually trained (letter-tokens vs word-outputs, single-token vs multi-token labels), **flag it back instead of guessing** — getting the logit readout right is load-bearing for the coverage guarantee.

---

## 8. Pointers to canonical source files

When you can pull this repo onto your platform, the must-read files in dependency order:

| File | Why |
|---|---|
| `code/outputs/sft/README.md` | SFT data format, label rule, train-time choices |
| `code/scripts/build_sft_router_v1.py` | Exact label-derivation code |
| `code/core/calibration.py` | Existing split-conformal scaffold |
| `code/core/conformal_layer.py` | Wrapper-style deployment pattern |
| `code/core/short_circuit_runner.py` | How cheapest-action labels come from real rollouts |
| `code/agents/code_agent.py` | Action prompt templates (proceed / reflect / replan / escalate) |
| `code/benchmarks/code_env.py` | Test runners — needed only if you re-run rollouts for fresh test data |
| `progress.md` (repo root) | Current data/rollout state |
| `code/outputs/bigcodebench/code_bigcodebench_gradient.md` | Long-form paper plan; check §"Direction & paper plan" |

If those aren't reachable on your platform, this doc plus the LlamaFactory checkpoint should be enough to start. The interface in §3 is the only thing that matters — once you have probability scores per problem, CRC is ~200 lines of numpy.

---

## 9. Known unknowns / risks to flag

- **🚨 Router quality has not been validated yet.** The checkpoint exists but no one has confirmed that the trained router (raw argmax, no CRC, no temperature scaling) **strictly beats** the trivial baselines on the cost-accuracy plane:
  - always-escalate (cost ≈ 12, fail rate ≈ 17% avg across datasets)
  - heuristic 2-tier proceed-then-escalate (cost ≈ 1 + 12 · p(fail) ≈ 7, fail ≈ 17%)
  - naive cascade (cost ≈ 17, fail ≈ 8%)

  **If the trained router doesn't dominate these baselines, CRC cannot save it.** CRC only slides along the achievable Pareto frontier; it cannot push the frontier further down-and-left. The frontier is fixed by router quality, which is fixed by SFT data + model size.

  At 1k SFT examples with 3-4 imbalanced classes (66% unsolvable in the 4cls variant), there is a real risk the router doesn't generalize. Mitigations:
  1. Re-train on 11k+ examples (Phase A finished, APPS in flight will add ~9k more — see `progress.md`)
  2. Use the Arch B 5cls variant (11k examples already) instead of 3cls (~2k)
  3. Try smaller cost-tied label collapse (3 classes: proceed / cheap-recovery / escalate)

  **Do this validation step before anything else** — it's the paper's life-or-death precondition.

- **Which exact SFT variant was trained?** Check the LlamaFactory training config for `dataset:` field — should be one of `router_no_reason_v1_3cls / _4cls / _archB_4cls / _archB_5cls`. This determines `K` (3, 4, or 5) and whether `proceed` is a valid output.
- **Output format**: the §3.2 score extraction assumes the router outputs a **single letter token** (`A` / `B` / `C` / `D`) at the answer position. The current SFT data (built by `code/scripts/build_sft_router_v1.py`) outputs **full action words** (`reflect`, `replan`, `escalate`). Verify what the checkpoint was actually trained on:
  - If letter tokens: §3.2 works as-is.
  - If full words: score the **first token of each action word** (e.g. `tokenizer.encode(' reflect')[0]`), and verify those first tokens are distinct across actions. If they collide (rare but possible), switch to multi-token scoring (sum log-probs over the action's token sequence) — less clean but still valid.
- **Cost vector inconsistency**: paper plan uses `proceed=1, reflect=1.2, replan=1.5, escalate=8`. The SFT label-building code uses `proceed=1, reflect=2, replan=2, escalate=13` (reflect and replan are cost-tied; the trained router was labelled with this assumption). Pick **one** vector and use it consistently for both the cost-regularized argmax and the figure axis; document the choice. The cost-tied 2/2 means λ effectively has no signal to separate reflect from replan — if you want to distinguish them, use the 1.2/1.5 version even though the SFT labels were built with 2/2.
- **Calibration set i.i.d. assumption**: if you use the SFT eval split as calibration, the guarantee is on the training distribution. To get coverage on APPS / out-of-distribution, calibrate on those datasets separately.
- **Finite-sample slack**: with calibration set of size n, the test-time Pr[escalate] or Pr[fail] can exceed the calibration-set empirical value by O(1/√n). With n = 500-1000 calibration examples, slack is ~3-5%. Either (a) bake the slack into the constraint (subtract from C / α before picking λ) or (b) report it explicitly in the validation table.
- **Outcomes matrix** (`cal_outcomes[i, a]`) needs all four `successful_actions` to be observed per problem. Phase A rollouts use a **short-circuit** runner that doesn't always run every action — if proceed passes, reflect/replan/escalate are never tried. For CRC calibration of all-four-action policies you must use the **independent** A/B/C/D sweeps.

  Current outcomes-complete inventory (across `outputs/<dataset>/v55/grad/`):
  - BCB n=30, LCB n=30, TACO ×4 tiers n=30 each, APPS n=30, MBPP n=30, CodeContests n=30, ClassEval n=30, HumanEvalFix n=30, DS-1000 n=30, SciCode n=30 — about **350-380 problems total** with all 4 actions independently observed.

  At n_cal ≈ 380, the Hoeffding slack at δ=0.05 is ε ≈ √(log(40)/760) ≈ **7%**. This means to enforce `Pr[escalate] ≤ 0.20` you'd actually require `\hat{Pr}_cal ≤ 0.13` — eating ~35% of the budget on calibration uncertainty alone. The achievable Pareto frontier with this n_cal is thin.

  **Recommended action**: run an additional **independent A/B/C/D sweep on 500-1000 random Phase A problems** before CRC calibration. This requires a small patch to `code/scripts/collect_rollouts_opt.py` (currently does short-circuit; needs a `--full-sweep` mode that runs all 4 actions per problem). Cost ~$15-25, wall ~6-8h. With n_cal ≈ 1000, slack drops to ~3-4%, enabling a real headline figure. **Without this, CRC's finite-sample story is too weak to be defensible at reviewing time.**

---

That's everything you need. Reach back here through your normal channel if you need clarifications on data shape or evaluation protocol; I can confirm against the live code.
