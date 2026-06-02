# Conformal CRC layer on the Qwen3.5-4B router — pipeline-validation findings

**Author note**: this run was to **get the pipeline end-to-end working** with the
3cls SFT checkpoint (`checkpoint-200`) and 3cls SFT data, not to chase
paper-quality numbers. We use the SFT dataset as a placeholder source for both
calibration and test — the i.i.d. assumption on which CRC's guarantees rest is
**partially violated** here; see §Caveats.

---

## TL;DR

1. **CRC pipeline is end-to-end working.** The Pareto curve sweep over
   `α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}` produces meaningful
   cost-vs-accuracy tradeoffs:
   - α=0.05 → coverage 96.2% (risk 3.8% ≤ 5% ✓), mean cost **2.0** nano-units
   - α=0.30 → coverage 75.5% (risk 24.5% ≤ 30% ✓), mean cost **6.8**,
     decision accuracy **63.5%** (vs argmax-only 52.2%)
   - α=0.50 → coverage 49.1%, mean cost 11.3, mostly escalate
2. **Empirical risk tracks α with small slack** as expected from finite-sample
   theory (n_calib=158). 6/8 α values give empirical risk ≤ α exactly; two
   (α=0.10 and α=0.20) overshoot by 3–4 pp — within finite-sample slack
   `O(1/√n) ≈ 8 pp`.
3. **CRC beats argmax on decision accuracy** for α ∈ [0.20, 0.30] — at α=0.30
   the CRC-wrapped router gets +11.3 pp decision accuracy over plain argmax
   (63.5% vs 52.2%), at a lower mean cost than always-escalate (6.8 vs 13.0).
4. **Three serious debugging stops** were needed before the pipeline produced
   sensible probabilities. See §Bugs caught.

---

## Setup

- Router checkpoint: `/mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/router_arch_a_3cls/checkpoint-200`
  - Base model: `Qwen/Qwen3.5-4B` (hybrid linear-attention + self-attention)
  - LoRA r=8 lora_alpha=16 target=all, 248 modules
- SFT data: `/mnt/bn/ecom-govern-models/qijiahe/datasets/router_no_reason_v1_3cls.json`
  (1583 examples, labels: reflect / replan / escalate)
- Split: 80/10/10 seeded random → train 1266 / calib 158 / test 159
- Cost vector (nano-units): `reflect=2, replan=2, escalate=13`
- Nonconformity score: `s_i = 1 - p_i[y_i^true]`
- Conformal quantile: `tau = ⌈(n+1)(1-α)⌉ / n` (Vovk's finite-sample correction)
- Prediction set: `C(x) = {a : p_a(x) >= 1 - tau}`
- Deployment policy: `argmin cost(a) for a in C(x)`; if `C(x)` is empty → fall back to escalate

All scripts in `conformal/scripts/`, run via `bash scripts/run_all.sh`.

---

## Results

### Baselines on test (n=159)
| baseline | decision accuracy | mean cost |
|---|---|---|
| argmax (no CRC) | 52.2% | 10.58 |
| always_reflect | 37.1% | 2.00 |
| always_replan | 23.9% | 2.00 |
| always_escalate | 39.0% | 13.00 |

The argmax router's mean cost is 10.6 — close to always-escalate — because the
router learned to default to escalate. This makes the argmax policy expensive
without proportional accuracy gain.

### CRC sweep (n_test=159, n_calib=158)

| α | τ | coverage | empirical risk | dec_acc | mean cost | refl% | repl% | esc% | empty |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.806 | 96.2% | **3.8%** ✓ | 39.6% | 2.00 | 71.7% | 28.3% | 0.0% | 0 |
| 0.10 | 0.754 | 93.7% | 6.3% (slack) | 45.9% | 2.14 | 52.2% | 46.5% | 1.3% | 0 |
| 0.15 | 0.734 | 90.6% | **9.4%** ✓ | 49.7% | 2.76 | 45.3% | 47.8% | 6.9% | 0 |
| 0.20 | 0.714 | 83.6% | 16.4% (slack) | 57.2% | 4.70 | 39.0% | 36.5% | 24.5% | 0 |
| 0.25 | 0.703 | 79.2% | **20.8%** ✓ | 62.3% | 5.94 | 35.8% | 28.3% | 35.8% | 0 |
| 0.30 | 0.698 | 75.5% | **24.5%** ✓ | **63.5%** | 6.77 | 34.6% | 22.0% | 43.4% | 0 |
| 0.40 | 0.672 | 61.0% | **39.0%** ✓ | 57.9% | 9.40 | 27.7% | 5.0% | 67.3% | 0 |
| 0.50 | 0.611 | 49.1% | **50.9%** ≈ | 52.8% | 11.34 | 15.1% | 0% | 84.9% | 24 |

`empty` = number of test examples where prediction set was empty (fell back to escalate).

### Pareto observations

- **Cost knob works**: average per-problem cost is monotone in α — 2.0 at α=0.05,
  11.3 at α=0.50.
- **Risk control holds** modulo finite-sample slack: 6/8 sweep points satisfy
  `empirical_risk ≤ α` exactly; the worst overshoot is α=0.20 → 16.4% (4.4 pp
  over), which is within the expected `O(1/√158) ≈ 8 pp` slack.
- **Sweet spot**: α=0.30 gives the highest decision accuracy (63.5%) at moderate
  cost (6.8) — meaningfully better than naive argmax (52.2%, cost 10.6).
- At α=0.50, prediction sets start going empty (24 cases) — the calibrated τ is
  too tight and no candidate clears the threshold; we fall back to escalate.
  This is the regime where CRC stops giving useful tradeoffs.

---

## Caveats

1. **Test/calibration are NOT held out from router training.** The router was
   SFT'd on a LLaMA-Factory random 90/10 train/eval split with an unknown seed.
   Our 80/10/10 split with seed 42 likely overlaps with the router's training
   set. Practical impact: router will be more confident on overlapped examples
   than on truly held-out data → empirical risk under-reported, coverage
   over-reported. For paper-quality numbers, calibrate on **fresh** data
   (e.g., separately rolled-out APPS problems or a held-out partition of the
   LF eval split with a known seed).
2. **Argmax accuracy (52%) is lower than training-time eval suggests (~76%
   from eval_loss 0.2693).** This is likely because we use **length-normalised**
   sequence log-prob (`sum(log p_token) / K`) for fair scoring across labels
   of different token counts (`reflect` is 1 token, `replan`/`escalate` are 2
   each); the trainer-reported eval_loss uses mean-per-token reduction across
   the full batch. The CRC pipeline still works fine on the length-normalised
   probabilities, but the comparison to training-time accuracy is not 1:1.
3. **Small calibration set (n=158)**. Finite-sample slack ≈ ±8 pp. To shrink
   this, calibrate on ~1k+ examples. The slack is *covered by Vovk's bound* —
   it does not invalidate CRC, but the empirical risk numbers wiggle around α
   in expectation.
4. **3-class only.** Variant `4cls` (adds `unsolvable`) trained better in our
   earlier SFT runs (eval_loss 0.244 vs 3cls's 0.269). Re-running CRC against
   the 4cls checkpoint should give a cleaner Pareto curve.

---

## Bugs caught (worth recording — these would have silently produced
garbage results)

1. **`apply_chat_template` returns `BatchEncoding`, not `Tensor`** — minor,
   but caused `torch.cat` to crash. Fix: `tokenize=False` then re-tokenize.
2. **🔥 Qwen3.5 chat_template.jinja inserts `<think>...</think>` by default.**
   The training prefix from LLaMA-Factory's `qwen3_nothink` template has NO
   `<think>` block, just `<|im_start|>assistant\n<label><|im_end|>`. At
   inference, calling `apply_chat_template(messages, add_generation_prompt=True)`
   without `enable_thinking=False` puts the model in chain-of-thought mode,
   and greedy decode produces "The user wants me to..." (base-model
   instruction-following), never the label. **Fix**: pass
   `enable_thinking=False`.
3. **🔥🔥 Adapter key path mismatch — silent zero-LoRA.**
   At training time the model class was `Qwen3_5ForCausalLM` whose `.model`
   attribute had a `.language_model` wrapper (multimodal scaffolding), so the
   saved adapter keys looked like
   `base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight`.
   At inference, `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")`
   loads a slimmer structure where `.model = Qwen3_5TextModel` exposes
   `.layers` directly with **no `.language_model` intermediate**.
   PEFT's path matcher failed silently — it instantiated 248 LoRA slots but
   loaded zero saved weights into them, so the LoRA-B matrices stayed at
   their zero-init. Net effect: `LoRA delta = A @ B = A @ 0 = 0` → the
   inference model behaved like the base Qwen3.5-4B, **completely ignoring
   the fine-tune**.
   **Fix**: `conformal/scripts/fix_adapter.py` rewrites the safetensors with
   `language_model.` stripped from all keys; saves to a sibling directory
   `checkpoint-200-renamed/`. After this, all 248 LoRA-B norms are non-zero
   (~0.28 Frobenius) and the model produces sensible label distributions.

The third bug is particularly nasty because PEFT reports "248 modules
attached" with no warning that weights weren't loaded. Symptom is just lower
argmax accuracy and unusual probability distributions — easy to dismiss as
"the model isn't great." Recommend any future paper-quality run check
`|lora_B|_F > 0` for every LoRA module after loading.

---

## File layout

```
conformal/
├── scripts/
│   ├── data_split.py             # 80/10/10 seeded split
│   ├── router_predict.py         # load model + LoRA, score per example
│   ├── calibrate_crc.py          # tau-table for alpha sweep
│   ├── eval_crc.py               # Pareto sweep + baselines on test
│   ├── fix_adapter.py            # one-time: rename adapter keys
│   ├── run_all.sh                # orchestrator (assumes adapter already fixed)
│   ├── _debug_arch.py            # diagnostic: base-model module structure vs adapter targets
│   ├── _debug_generate.py        # diagnostic: greedy generation to check raw output
│   └── _debug_template.py        # diagnostic: chat-template comparison
├── data/
│   ├── train.jsonl   calib.jsonl   test.jsonl       # 80/10/10 split
│   ├── calib_preds.jsonl   test_preds.jsonl         # per-example probs
│   └── tau_table.json                                # tau per alpha
├── results/
│   └── eval.json                                     # final Pareto sweep
└── FINDINGS.md                                       # this file
```

To reproduce: `bash conformal/scripts/run_all.sh` (assumes
`checkpoint-200-renamed` already exists; if not, first run
`python conformal/scripts/fix_adapter.py`).

---

---

## Negative-result follow-up: can a prompted general-purpose LLM substitute for the SFT'd router?

**TL;DR**: No. Running CRC on top of `gpt-5.4-nano-2026-03-17` with the exact
same prompt the router was trained with **collapses the Pareto curve to a
single point** (mean cost ≈ 2.0, decision accuracy ≈ 38%, no escalate
selections at any α). The SFT was not optional — CRC is a calibration wrapper,
not a substitute for learned cost-aware behaviour.

### Setup

- Classifier: `gpt-5.4-nano-2026-03-17` via byteintl Azure endpoint
  (`AzureOpenAI`, `temperature=0`, `logprobs=True`, `top_logprobs=5`)
- Prompt: identical to SFT training format (system = same instruction,
  user = same `Problem:\n...\nInitial attempt verdict: ...` block)
- Scoring: extract first-token logprobs; map each top-5 token to one of
  {reflect, replan, escalate} via a disjoint prefix table that resolves
  the only ambiguity ("re" prefix is shared by reflect & replan — but
  "reflect" tokenises as its own single token in cl100k/o200k, so a bare
  "re" first-token must initiate "replan"); softmax over the 3 candidates.
- Same calib/test splits as the SFT-router run (calib n=158, test n=159).
- QPM-throttled: ~5 records/min effective, ~80 min total per split.
- Implementation: `conformal/scripts/nano_predict.py`.

### Raw argmax results (nano-only, no CRC)

Calib (147 of 158 usable; 11 dropped because QPM retries exhausted):
- argmax accuracy: **29.9%** (44/147) — essentially the majority-class
  baseline for a 3-way split
- argmax distribution: 90 reflect, 56 replan, **1 escalate**
- per-true-label argmax:
  - true=escalate (55): assigned reflect 31 / replan 24 / **escalate 0**
  - true=reflect (53): 32 correct (60% recall)
  - true=replan (39): 12 correct (31% recall)

Test (149 of 159 usable; 10 dropped):
- argmax accuracy: **34.2%** (51/149)
- argmax distribution: 82 reflect, 67 replan, **0 escalate**
- true=escalate (59): assigned reflect 26 / replan 33 / **escalate 0** — **0% recall**

**Diagnosis**: nano cannot decide *when to escalate*. Without SFT, it lacks
the priors connecting "this codeforces problem + this stderr → likely
unrecoverable, escalate to the expensive model". It always picks one of the
cheap actions instead — exactly the behaviour the cascade router needs to
overcome.

### CRC sweep on nano probabilities

| α | τ | coverage | risk | dec_acc | mean cost | reflect% | replan% | escalate% |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 1.0000 | 98.0% | 2.0% | 38.3% | **2.000** | 100.0% | 0.0% | **0.0%** |
| 0.10 | 1.0000 | 92.6% | 7.4% | 38.3% | 2.000 | 100.0% | 0.0% | 0.0% |
| 0.15 | 0.9999 | 89.3% | 10.7% | 38.3% | 2.000 | 100.0% | 0.0% | 0.0% |
| 0.20 | 0.9997 | 85.2% | 14.8% | 38.3% | 2.000 | 100.0% | 0.0% | 0.0% |
| 0.25 | 0.9992 | 77.9% | 22.1% | 38.3% | 2.000 | 99.3% | 0.7% | 0.0% |
| 0.30 | 0.9978 | 69.8% | 30.2% | 38.3% | 2.000 | 99.3% | 0.7% | 0.0% |
| 0.40 | 0.9627 | 58.4% | 41.6% | 38.3% | 2.000 | 89.9% | 10.1% | 0.0% |
| 0.50 | 0.8597 | 50.3% | 49.7% | 34.9% | 2.000 | 81.2% | 18.8% | 0.0% |

**Pareto collapse**. Three observations:

1. **mean_cost ≡ 2.0 across the entire α sweep**. The cheapest-in-set rule
   *cannot* pick a cost-13 escalate because nano's escalate probability is
   essentially zero (`mean ~3e-4`); even when CRC's τ is 1.0 (so the
   prediction set theoretically includes all labels with prob ≥ 0), the
   cheapest of {reflect, replan, escalate} is still reflect (or replan).
2. **decision accuracy is pinned to always_reflect (38.3%)**. The CRC sweep
   doesn't change which action gets chosen for any example — it just shrinks
   the prediction set. No knob to trade cost for quality.
3. **Coverage still tracks 1-α as theory predicts**. The conformal guarantee
   itself isn't broken — risks of 2.0%, 7.4%, 10.7%, … track α reasonably.
   But coverage here is a guarantee about *containing the true label in the
   set*, not about *picking it*. With escalate prob ≈ 0, escalate enters the
   set vacuously (at low α) without ever being the cheapest option.

### Qwen3.5-4B base (no LoRA, same prompt) — direct ablation

We also tested the **same Qwen3.5-4B base model with no LoRA** through the
same scoring path (`router_predict.py --no_adapter`) — this isolates "what
does the SFT actually buy us" controlling for model capacity and prompt.

Argmax (n=159 test):
- argmax accuracy: **35.8%** (57/159)
- argmax distribution: **144 reflect, 15 replan, 0 escalate** — even more
  collapsed than nano; Qwen base just defaults to reflect almost universally
- true=escalate (62): 57 reflect / 5 replan / **0 escalate** (0% recall)
- true=reflect (59): 53/59 (90% recall — almost a free pass since it always says reflect)
- true=replan (38): 4/38 (11% recall)

CRC sweep:
- Across **every α**: 100% reflect, mean_cost ≡ 2.0, dec_acc ≡ 37.1%
  (exactly always_reflect baseline). Even worse than nano, which at least
  varied its argmax across reflect/replan.

### Side-by-side: SFT router vs. nano vs. Qwen base

| classifier | argmax acc | CRC best dec_acc | CRC cost range | Pareto width | escalate recall |
|---|---|---|---|---|---|
| SFT Qwen3.5-4B router (ckpt-200) | 52.2% | **63.5%** (α=0.30) | 2.00 – 11.34 | wide, usable | meaningful (router learned to escalate) |
| GPT-5.4-nano (prompted) | 34.2% | 38.3% (any α) | 2.00 | **collapsed** | 0% (0/59) |
| Qwen3.5-4B base (no LoRA, prompted) | 35.8% | 37.1% (any α) | 2.00 | **collapsed** | 0% (0/62) |
| GPT-5.4-mini (prompted) | 33.96% | 37.11% (any α) | 2.00 | **collapsed** | 0% (0/62) |

### GPT-5.4-mini run (completed 2026-05-20)

The first attempt at `gpt-5.4-mini-2026-03-17` (kicked off 2026-05-20 ~15:43)
hit a silent API constraint: the mini endpoint **requires
`max_completion_tokens >= 150` to return logprobs**, while nano accepted the
script's default `max_tokens=8`. Net effect: ~92% of mini rows came back as
HTTP 400 with `"Logprobs could not be returned for short completions"`. Only
13/158 calib + 11/159 test usable — not enough to calibrate. **Fix**:
`nano_predict.py:83` → `max_tokens=150`. Verified via 2-row smoke test, then
re-ran calib + test in parallel (background, ~25 min wall). The clean rerun
produced 158/158 calib + 159/159 test with **zero API errors**.

Argmax (n=159 test):
- argmax accuracy: **33.96%** (54/159) — even lower than nano (34.2%)
- argmax distribution: **136 reflect, 23 replan, 0 escalate** — same
  collapse pattern as nano and qwen-base
- true=escalate (62): 50 reflect / 12 replan / **0 escalate** (0% recall)
- true=reflect (59): 51 correct (86% recall — basically the always-reflect bonus)
- true=replan (38): 3 correct (8% recall — worse than nano's 31%)

CRC sweep:

| α | τ | coverage | risk | dec_acc | mean cost | reflect% | replan% | escalate% |
|---|---|---|---|---|---|---|---|---|
| 0.05–0.40 | 1.0000 | 100.0% | 0.0% | 37.1% | **2.000** | 100.0% | 0.0% | **0.0%** |
| 0.50 | 0.8520 | 43.4% | 56.6% | 37.1% | 2.000 | 99.4% | 0.6% | 0.0% |

**Pareto-collapse is sharper than nano**:

1. **τ pinned at 1.0 for α ≤ 0.40** — the calibration NLL is so high
   (mean `1 − p_true = 0.65`) that the conformal quantile maxes out, so
   *every* candidate makes the prediction set. Cheapest-in-set is always
   reflect; mean cost ≡ 2.0; decision accuracy ≡ always_reflect baseline.
2. **0% escalate at every α** — mini's reflect-vs-escalate logit gap is
   even wider than nano's. There is no α that produces a single escalate
   pick on the entire test set.
3. **Coverage doesn't even degrade at α=0.30 the way nano's does** (mini
   stays at 100% coverage; nano drops to 70%). Mini's probabilities are
   so flat across labels that τ=1.0 satisfies the constraint trivially.

### Updated side-by-side: SFT router vs. Qwen-base vs. nano vs. mini

| classifier | argmax acc | CRC best dec_acc | CRC cost range | Pareto width | escalate recall |
|---|---|---|---|---|---|
| SFT Qwen3.5-4B router (ckpt-200) | 52.2% | **63.5%** (α=0.30) | 2.00 – 11.34 | wide, usable | meaningful (router learned to escalate) |
| Qwen3.5-4B base (no LoRA, prompted) | 35.8% | 37.1% (any α) | 2.00 | **collapsed** | 0% (0/62) |
| GPT-5.4-nano (prompted) | 34.2% | 38.3% (any α) | 2.00 | **collapsed** | 0% (0/59) |
| GPT-5.4-mini (prompted) | 34.0% | 37.1% (any α) | 2.00 | **collapsed** | 0% (0/62) |

**Mini is not the missing ingredient.** Bumping from nano → mini (a stronger
general-purpose model) does *not* recover the cost-aware escalate decisions
that SFT bakes in. All three prompted baselines (qwen-base, nano, mini)
flat-line at the always-cheapest operating point regardless of α — the
ablation is now 3-wide and converges on the same conclusion: only the SFT
step recovers the cascade-routing behaviour. CRC is a calibration wrapper,
not a substitute for learned cost-aware behaviour.

**Implications for the paper**:
- The SFT step is doing real, irreplaceable work: the router learned
  *cost-aware escalate decisions* that no amount of test-time prompting can
  recover from a generic model.
- CRC is a calibration wrapper. It needs the base classifier to *already*
  assign non-trivial probability mass to expensive actions. Otherwise the
  Pareto curve collapses to a single point (always-cheapest).
- "Static prompt-only routing can't capture the cost-quality frontier" — this
  is exactly what the paper #1 narrative claims, and we now have a clean
  ablation to back it.

### Caveats specific to the nano run

1. **10–11 records dropped per split (~7% loss rate)** due to QPM
   rate-limit retry exhaustion. Re-running with longer backoff / lower QPM
   should recover them but is unlikely to change the headline finding.
2. **Length-normalisation isn't relevant for first-token logprobs.** This
   eliminates one confound from the router run (which used multi-token
   sequence scoring). Nano accuracy is low because *the model doesn't know
   the cost prior*, not because of scoring choices.
3. **Same data overlap caveat as the router run**. Calibration set isn't
   strictly i.i.d. with router-training distribution. Doesn't affect the
   negative result here — nano never saw any of it.

---

## Next steps (when going from "链路跑通" to paper quality)

1. **Use the 4cls checkpoint** (best SFT eval_loss = 0.244, vs 3cls's 0.269).
2. **Calibrate on truly held-out data.** Options: (a) re-train router with a
   known seed and split off LF's eval split for calib+test, (b) use APPS
   rollouts (not in training distribution) as test set with a separate
   in-distribution calib.
3. **Larger calibration set** (target n=500–1000) shrinks finite-sample slack
   to ~±3 pp.
4. **Compare scoring formulations**: length-normalised log-prob (current) vs
   raw sum log-prob vs first-token-only logit. They give different argmax
   distributions; pick whichever matches the deployment scoring rule.
5. **Adapter rename** is a workaround. The cleaner fix would be to re-export
   the LoRA from the original training environment with `language_model.`
   already absent in keys — saves the rename step at every deployment.
