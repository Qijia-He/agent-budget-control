"""eval_crc v2 — λ-CRC over cost-regularized argmax, with two cost modes.

Differences vs the original eval_crc.py:
  - Implements the HANDOFF §3.4 framing: policy_λ(x) = argmax_a [p_a − λ·cost(a)]
    (cost-regularized argmax). The original used τ-prediction-set + cheapest-in-set,
    which the HANDOFF explicitly says NOT to use for paper #1.
  - Sweeps λ over candidate breakpoints (analytic step-function frontier).
  - Two cost modes:
      --cost_mode unit   → {reflect:2, replan:2, escalate:13} (nano-units)
      --cost_mode tokens → per-problem $-cost loaded from action_costs_usd.json
                            (averaged over test for the policy's cost vector,
                             then per-problem reported costs use that problem's
                             actual $-cost).
  - Reports per-problem cost INCLUDING the sunk proceed call (cost=1 in unit
    mode, proceed_usd in tokens mode).
  - Same metric set as v1: dec_acc (chosen == true cheapest label), chosen_dist,
    avg cost, plus the always_* baselines and argmax-only baseline.

NOTE on success metric: we don't have per-(problem, action) PASS observations
for the SFT 3cls calib/test split (those would require outcomes_n30 join, and
the SFT-derived calib set doesn't overlap with outcomes_n30 cleanly). So we
report decision_accuracy as a proxy (matches v1's framing). Adding true
solve_rate is a separate follow-up.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np


ACTIONS = ["reflect", "replan", "escalate"]
COST_UNIT = {"reflect": 2.0, "replan": 2.0, "escalate": 13.0}
PROCEED_COST_UNIT = 1.0


def load_preds(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out.append(r)
    return out


def load_action_costs(path: Path) -> dict:
    return json.load(path.open())


def build_cost_vector(records: List[dict], mode: str, action_costs: dict):
    """Return:
        cost_vec  ∈ R^3 — vector used in λ·cost(a) (recovery-only)
        per_problem_recovery_cost  shape [N, 3] — per problem per action $ recovery
        proceed_cost_per_problem  shape [N]
    """
    if mode == "unit":
        cv = np.array([COST_UNIT[a] for a in ACTIONS])
        n = len(records)
        per_p_recov = np.broadcast_to(cv, (n, 3)).copy()
        proceed_arr = np.full(n, PROCEED_COST_UNIT)
        # cost vector used by policy: the global (action-unit) one.
        return cv, per_p_recov, proceed_arr

    # tokens mode
    pp = action_costs["per_problem"]
    n = len(records)
    per_p_recov = np.zeros((n, 3))
    proceed_arr = np.zeros(n)
    for i, r in enumerate(records):
        key = f"{r['dataset']}::{r['problem_id']}"
        info = pp[key]
        per_p_recov[i, 0] = info["reflect_usd"]
        per_p_recov[i, 1] = info["replan_usd"]
        per_p_recov[i, 2] = info["escalate_usd"]
        proceed_arr[i] = info["proceed_usd"]
    # Cost vector for the policy: use the dataset-wide MEAN of each recovery
    # action's cost across the calibration set. This makes policy_λ a global
    # cost-aware rule that doesn't peek at per-problem cost (cleaner from a
    # CRC theory standpoint — λ is a single scalar over the deployment dist).
    cv = per_p_recov.mean(axis=0)
    return cv, per_p_recov, proceed_arr


def probs_array(records: List[dict]) -> np.ndarray:
    P = np.zeros((len(records), 3))
    for i, r in enumerate(records):
        for j, a in enumerate(ACTIONS):
            P[i, j] = r["probs"][a]
    return P


def true_labels(records: List[dict]) -> np.ndarray:
    return np.array([ACTIONS.index(r["true_label"]) for r in records])


def policy_lambda(probs: np.ndarray, cost_vec: np.ndarray, lam: float) -> np.ndarray:
    """Vectorized: returns chosen action idx per row."""
    return np.argmax(probs - lam * cost_vec, axis=-1)


def enumerate_breakpoints(probs: np.ndarray, cost_vec: np.ndarray) -> List[float]:
    """λ where two actions tie:  p_a − λ·c_a = p_b − λ·c_b
       → λ = (p_a − p_b) / (c_a − c_b)
    """
    breakpoints = set()
    N, K = probs.shape
    for i in range(N):
        for a in range(K):
            for b in range(K):
                if a >= b:
                    continue
                if cost_vec[a] == cost_vec[b]:
                    continue
                num = probs[i, a] - probs[i, b]
                den = cost_vec[b] - cost_vec[a]
                lam = num / den
                if lam > 0:
                    breakpoints.add(float(lam))
    return sorted(breakpoints)


def metrics_at_lam(probs, y, cv, per_p_recov, proceed_arr, lam: float):
    chosen = policy_lambda(probs, cv, lam)
    # per-problem total cost = proceed + chosen recovery (using REAL per-problem
    # cost, not the policy's global cost vector — this is the operating-cost
    # we'd actually pay)
    total_cost = proceed_arr + per_p_recov[np.arange(len(chosen)), chosen]
    dec_acc = float((chosen == y).mean())
    chosen_dist = {
        ACTIONS[i]: int((chosen == i).sum()) for i in range(3)
    }
    return {
        "n": len(chosen),
        "lam": float(lam),
        "mean_cost": float(total_cost.mean()),
        "p50_cost": float(np.median(total_cost)),
        "decision_accuracy": dec_acc,
        "chosen_dist": chosen_dist,
        "pr_escalate": float((chosen == 2).mean()),
    }


def evaluate_baselines(probs, y, cv, per_p_recov, proceed_arr):
    """Argmax (no CRC), always_reflect / always_replan / always_escalate."""
    out = []
    # argmax (no CRC)
    chosen = np.argmax(probs, axis=-1)
    total = proceed_arr + per_p_recov[np.arange(len(chosen)), chosen]
    out.append({
        "name": "argmax",
        "n": len(chosen),
        "decision_accuracy": float((chosen == y).mean()),
        "mean_cost": float(total.mean()),
        "chosen_dist": {ACTIONS[i]: int((chosen == i).sum()) for i in range(3)},
    })
    for j, a in enumerate(ACTIONS):
        chosen = np.full(len(y), j)
        total = proceed_arr + per_p_recov[:, j]
        out.append({
            "name": f"always_{a}",
            "n": len(chosen),
            "decision_accuracy": float((y == j).mean()),
            "mean_cost": float(total.mean()),
            "chosen_dist": {ACTIONS[i]: (len(y) if i == j else 0) for i in range(3)},
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib_preds", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/calib_preds_with_pid.jsonl")
    p.add_argument("--test_preds", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/test_preds_with_pid.jsonl")
    p.add_argument("--action_costs", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/action_costs_usd.json")
    p.add_argument("--cost_mode", choices=["unit", "tokens"], required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max_lams", type=int, default=64,
                   help="downsample the breakpoint sweep to this many evenly-spaced points")
    args = p.parse_args()

    out_path = args.output or f"/mnt/bn/ecom-govern-models/qijiahe/conformal/results/eval_v2_{args.cost_mode}.json"

    calib = load_preds(Path(args.calib_preds))
    test = load_preds(Path(args.test_preds))
    action_costs = load_action_costs(Path(args.action_costs))
    print(f"loaded calib={len(calib)} test={len(test)}")
    label_dist = {}
    for r in test:
        label_dist[r["true_label"]] = label_dist.get(r["true_label"], 0) + 1
    print(f"test label dist: {label_dist}")

    # build cost vectors + per-problem recovery cost (one for calib, one for test)
    cv_cal, ppc_cal, pa_cal = build_cost_vector(calib, args.cost_mode, action_costs)
    cv_test, ppc_test, pa_test = build_cost_vector(test, args.cost_mode, action_costs)
    # IMPORTANT: λ-policy uses ONE cost vector. By convention use the calib one
    # (operator-side knowledge). Apply it on test.
    cv = cv_cal
    print(f"\ncost_mode={args.cost_mode}")
    print(f"policy cost vector (cv) = " + ", ".join(f"{a}={cv[i]:.5f}" for i, a in enumerate(ACTIONS)))
    print(f"avg proceed cost on calib = {pa_cal.mean():.5f}, on test = {pa_test.mean():.5f}")

    probs_cal = probs_array(calib)
    probs_test = probs_array(test)
    y_cal = true_labels(calib)
    y_test = true_labels(test)

    # Build sweep λ from calibration breakpoints
    bps = enumerate_breakpoints(probs_cal, cv)
    print(f"\ncalib breakpoints: {len(bps)} (range [{bps[0]:.4g}, {bps[-1]:.4g}])" if bps else "no breakpoints")
    # add lam=0 anchor and downsample for clean table
    bps = [0.0] + bps + [bps[-1] * 1.5] if bps else [0.0, 1.0]
    if len(bps) > args.max_lams:
        sel = np.linspace(0, len(bps) - 1, args.max_lams).round().astype(int)
        bps_sweep = [bps[i] for i in sorted(set(sel))]
    else:
        bps_sweep = bps
    # use midpoints between adjacent breakpoints so the argmax lands cleanly inside an interval
    sweep_lams = []
    for i in range(len(bps_sweep) - 1):
        sweep_lams.append((bps_sweep[i] + bps_sweep[i + 1]) / 2)
    sweep_lams = [0.0] + sweep_lams + [bps_sweep[-1] * 1.5]
    sweep_lams = sorted(set(sweep_lams))
    print(f"swept λ values: {len(sweep_lams)}")

    # ----- BASELINES (on test) -----
    base = evaluate_baselines(probs_test, y_test, cv, ppc_test, pa_test)
    print("\n=== baselines (test) ===")
    for b in base:
        print(f"  {b['name']:18s}  dec_acc={b['decision_accuracy']:.4f}  "
              f"mean_cost={b['mean_cost']:.5f}  chosen={b['chosen_dist']}")

    # ----- λ SWEEP (on test) -----
    rows = []
    for lam in sweep_lams:
        m = metrics_at_lam(probs_test, y_test, cv, ppc_test, pa_test, lam)
        rows.append(m)

    print("\n=== λ sweep (test, cost_mode={}) ===".format(args.cost_mode))
    print(f"{'lambda':>10s} {'dec_acc':>8s} {'mean_cost':>11s} {'p50':>9s} "
          f"{'pr_esc':>8s}  refl%  repl%  esc%")
    # dedupe consecutive rows where chosen is identical (step function plateaus)
    last = None
    distinct = []
    for r in rows:
        sig = (r["chosen_dist"]["reflect"], r["chosen_dist"]["replan"], r["chosen_dist"]["escalate"])
        if sig != last:
            distinct.append(r)
            last = sig
    for r in distinct:
        cd = r["chosen_dist"]; n = r["n"]
        print(f"{r['lam']:10.4g} {r['decision_accuracy']:8.4f} {r['mean_cost']:11.5f} "
              f"{r['p50_cost']:9.5f} {r['pr_escalate']:8.3f}  "
              f"{cd['reflect']/n*100:5.1f}  {cd['replan']/n*100:5.1f}  {cd['escalate']/n*100:5.1f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "cost_mode": args.cost_mode,
            "cost_vec": {a: float(cv[i]) for i, a in enumerate(ACTIONS)},
            "proceed_cost_calib": float(pa_cal.mean()),
            "proceed_cost_test": float(pa_test.mean()),
            "baselines": base,
            "lambda_sweep_distinct": distinct,
            "lambda_sweep_all": rows,
        }, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
