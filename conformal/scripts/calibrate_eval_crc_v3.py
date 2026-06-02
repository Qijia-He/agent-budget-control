"""eval_crc v3 — REAL CRC with two modes and two cost forms.

Two CRC modes:
  Mode A (cost-budget):   pick smallest λ on calib s.t. mean_cost(λ) ≤ C - ε
                          → guarantee:  E_test[cost(chosen)] ≤ C   w.p. ≥ 1-δ
  Mode B (fail-budget):   pick largest λ on calib s.t. fail_rate(λ) ≤ α - ε
                          (monotonicity-corrected: largest λ such that
                           fail_rate(λ') ≤ α-ε ∀ λ' ≥ λ in sweep)
                          → guarantee:  P_test[fail] ≤ α   w.p. ≥ 1-δ
                          equivalently: P(test acc ≥ 1-α) ≥ 1-δ.

Two cost forms (cli flag --cost_mode):
  api_call: nano-units {proceed=1, reflect=2, replan=2, escalate=13}
  token:    per-problem $-cost from action_costs_usd.json (real OpenAI billing)

Finite-sample slack:
  Mode A: ε_cost = (c_max - c_min) · sqrt(log(2/δ) / (2 n_cal))  [Hoeffding]
  Mode B: ε_fail = sqrt(log(2/δ) / (2 n_cal))                    [Hoeffding]

Per-problem cost = proceed cost (sunk, always paid) + chosen recovery cost.
Per-problem fail = 1 if chosen recovery action is NOT in successful_actions.

NOTE on fail-rate conservatism: rollout is short-circuit, so escalate's pass
status is only observed when reflect=0 AND replan=0 (≈ 39% of problems).
For 'easy' problems (reflect=1 OR replan=1, ~61%) we don't actually know
whether escalate would have passed; we conservatively count it as 0 in the
matrix. This OVER-counts router-escalate-on-easy-problems as failures,
making Mode B's empirical fail rate biased upward. The guarantee direction
is preserved (test-time fail rate ≤ α still holds), with strictly more
slack than the calibration suggests. The headline cost-saving understated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


ACTIONS_3 = ["reflect", "replan", "escalate"]
ACTIONS_4 = ["reflect", "replan", "escalate", "unsolvable"]
COST_API_3 = {"reflect": 2.0, "replan": 2.0, "escalate": 13.0}
COST_API_4 = {"reflect": 2.0, "replan": 2.0, "escalate": 13.0, "unsolvable": 0.0}
PROCEED_API = 1.0

# Set by main() based on --num_classes; modules below read these globals.
ACTIONS = None
COST_API = None


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_arrays(records, action_costs, success_matrix, cost_mode):
    """Returns:
        P: [N,K] router probs
        Y: [N] true label idx
        O: [N,K] success outcome per action
        cv: [K] policy-side cost vector
        recov_cost_per_problem: [N,K] per-(problem,action) recovery cost
        proceed_cost_per_problem: [N] per-problem proceed cost
    K = len(ACTIONS), set globally by main().
    """
    K = len(ACTIONS)
    N = len(records)
    P = np.zeros((N, K))
    Y = np.zeros(N, dtype=int)
    O = np.zeros((N, K), dtype=int)
    recov = np.zeros((N, K))
    proceed_arr = np.zeros(N)

    for i, r in enumerate(records):
        for j, a in enumerate(ACTIONS):
            P[i, j] = r["probs"][a]
        Y[i] = ACTIONS.index(r["true_label"])
        key = f"{r['dataset']}::{r['problem_id']}"
        sm = success_matrix[key]
        for j, a in enumerate(ACTIONS):
            O[i, j] = int(sm[a])
        ac = action_costs["per_problem"][key]
        if cost_mode == "api_call":
            recov[i] = [COST_API[a] for a in ACTIONS]
            proceed_arr[i] = PROCEED_API
        else:  # token
            for j, a in enumerate(ACTIONS):
                recov[i, j] = ac.get(f"{a}_usd", 0.0)
            proceed_arr[i] = ac["proceed_usd"]

    cv = recov.mean(axis=0)
    return P, Y, O, cv, recov, proceed_arr


def policy(P, cv, lam):
    return np.argmax(P - lam * cv, axis=-1)


def enumerate_breakpoints(P, cv):
    bps = set()
    N, K = P.shape
    for i in range(N):
        for a in range(K):
            for b in range(a + 1, K):
                if cv[a] == cv[b]:
                    continue
                lam = (P[i, a] - P[i, b]) / (cv[b] - cv[a])
                if lam > 0:
                    bps.add(float(lam))
    return sorted(bps)


def hoeffding_eps(n, delta):
    return math.sqrt(math.log(2.0 / delta) / (2 * n))


# --------------- Mode A (cost-budget) ---------------

def calibrate_mode_A(P_cal, recov_cal, proceed_cal, cv, lams, C, delta):
    """smallest λ s.t. mean_cost(λ) ≤ C - ε on calib.
    Cost range for ε: total cost varies in [proceed+min(recov), proceed+max(recov)].
    """
    n = len(proceed_cal)
    # Hoeffding slack uses cost range (bounded variable)
    c_min = proceed_cal.min() + recov_cal.min(axis=1).min()
    c_max = proceed_cal.max() + recov_cal.max(axis=1).max()
    eps = (c_max - c_min) * hoeffding_eps(n, delta)
    target = C - eps

    # mean_cost(λ) is non-increasing in λ (each row's cost is non-increasing,
    # so the mean is too). Walk λ from smallest → largest, return first that satisfies.
    chosen_lam = None
    cal_mean = None
    for lam in [0.0] + list(lams):
        idx = policy(P_cal, cv, lam)
        cost = proceed_cal + recov_cal[np.arange(n), idx]
        mean = float(cost.mean())
        if mean <= target:
            chosen_lam = lam
            cal_mean = mean
            break
    return {
        "lam_hat": chosen_lam,
        "C": C,
        "eps": eps,
        "target_after_slack": target,
        "cal_mean_cost": cal_mean,
        "n_cal": n,
        "delta": delta,
        "feasible": chosen_lam is not None,
    }


# --------------- Mode B (fail-budget) ---------------

def calibrate_mode_B(P_cal, O_cal, cv, lams, alpha, delta):
    """largest λ s.t. fail_rate(λ') ≤ α - ε for all λ' ≥ λ in the sweep.
    (Monotonicity-corrected — fail_rate is NOT strictly monotone in λ.)
    """
    n = O_cal.shape[0]
    eps = hoeffding_eps(n, delta)
    target = alpha - eps
    lam_sweep = [0.0] + list(lams)
    fail_rates = []
    for lam in lam_sweep:
        idx = policy(P_cal, cv, lam)
        fail = 1 - O_cal[np.arange(n), idx]
        fail_rates.append(float(fail.mean()))
    # Walk from largest λ to smallest. Find LARGEST λ such that fail(λ')
    # ≤ target for ALL λ' ≥ λ. max_fail_seen accumulates the max over
    # [current_lam, max_lam]; first valid we encounter (= largest) is kept.
    chosen_lam = None
    cal_fail = None
    max_fail_seen = -1.0
    for lam, fr in reversed(list(zip(lam_sweep, fail_rates))):
        max_fail_seen = max(max_fail_seen, fr)
        if max_fail_seen > target:
            break  # all smaller λ also violate
        if chosen_lam is None:
            chosen_lam = lam
            cal_fail = fr
            # don't break — we want to verify the corrected rule holds by walking
            # through, but we keep the LARGEST λ (this one)
    return {
        "lam_hat": chosen_lam,
        "alpha": alpha,
        "eps": eps,
        "target_after_slack": target,
        "cal_fail_rate": cal_fail,
        "n_cal": n,
        "delta": delta,
        "feasible": chosen_lam is not None,
    }


# --------------- Test-time evaluation ---------------

def eval_lam_on_test(P_test, Y_test, O_test, recov_test, proceed_test, cv, lam):
    K = len(ACTIONS)
    n = len(Y_test)
    idx = policy(P_test, cv, lam)
    cost = proceed_test + recov_test[np.arange(n), idx]
    fail = 1 - O_test[np.arange(n), idx]
    chosen_dist = {ACTIONS[i]: int((idx == i).sum()) for i in range(K)}
    esc_idx = ACTIONS.index("escalate") if "escalate" in ACTIONS else None
    return {
        "lam": float(lam),
        "n": n,
        "mean_cost": float(cost.mean()),
        "p50_cost": float(np.median(cost)),
        "fail_rate": float(fail.mean()),
        "solve_rate": float(1.0 - fail.mean()),
        "dec_acc": float((idx == Y_test).mean()),
        "chosen_dist": chosen_dist,
        "pr_escalate": float((idx == esc_idx).mean()) if esc_idx is not None else None,
    }


def baselines(P_test, Y_test, O_test, recov_test, proceed_test):
    K = len(ACTIONS)
    out = []
    n = P_test.shape[0]
    idx = np.argmax(P_test, axis=-1)
    out.append({
        "name": "router_argmax",
        "mean_cost": float((proceed_test + recov_test[np.arange(n), idx]).mean()),
        "fail_rate": float((1 - O_test[np.arange(n), idx]).mean()),
        "solve_rate": float(O_test[np.arange(n), idx].mean()),
        "dec_acc": float((idx == Y_test).mean()),
        "chosen_dist": {ACTIONS[i]: int((idx == i).sum()) for i in range(K)},
    })
    for j, a in enumerate(ACTIONS):
        idx = np.full(n, j)
        out.append({
            "name": f"always_{a}",
            "mean_cost": float((proceed_test + recov_test[:, j]).mean()),
            "fail_rate": float((1 - O_test[:, j]).mean()),
            "solve_rate": float(O_test[:, j].mean()),
            "dec_acc": float((Y_test == j).mean()),
            "chosen_dist": {ACTIONS[i]: (n if i == j else 0) for i in range(K)},
        })
    return out


# --------------- main ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cost_mode", choices=["api_call", "token"], required=True)
    p.add_argument("--mode", choices=["A", "B", "both"], default="both")
    p.add_argument("--num_classes", type=int, choices=[3, 4], default=3)
    p.add_argument("--calib_preds", default=None,
                   help="default depends on num_classes")
    p.add_argument("--test_preds", default=None)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--C_sweep", default="3,4,5,6,8,10,12",
                   help="api_call mode: budget on E[cost] (units); we also try token-equivalents below")
    p.add_argument("--C_sweep_token", default="0.003,0.004,0.005,0.006,0.008,0.010",
                   help="token mode: budget on E[cost] ($)")
    p.add_argument("--alpha_sweep", default="0.10,0.20,0.30,0.40,0.50,0.60")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # bind ACTIONS / COST_API per --num_classes
    global ACTIONS, COST_API
    ACTIONS = ACTIONS_4 if args.num_classes == 4 else ACTIONS_3
    COST_API = COST_API_4 if args.num_classes == 4 else COST_API_3

    DATA = Path("/mnt/bn/ecom-govern-models/qijiahe/conformal/data")
    suffix = "_4cls" if args.num_classes == 4 else ""
    calib_path = Path(args.calib_preds) if args.calib_preds else DATA / f"calib_preds{suffix}_with_pid.jsonl"
    test_path = Path(args.test_preds) if args.test_preds else DATA / f"test_preds{suffix}_with_pid.jsonl"
    cal_recs = load_jsonl(calib_path)
    test_recs = load_jsonl(test_path)
    action_costs = json.load((DATA / "action_costs_usd.json").open())
    success_matrix = json.load((DATA / "success_matrix.json").open())

    P_cal, Y_cal, O_cal, cv, recov_cal, proceed_cal = build_arrays(
        cal_recs, action_costs, success_matrix, args.cost_mode)
    P_test, Y_test, O_test, _, recov_test, proceed_test = build_arrays(
        test_recs, action_costs, success_matrix, args.cost_mode)
    # IMPORTANT: cv computed on CALIB only (policy uses operator-known calib distribution)

    print(f"cost_mode={args.cost_mode}  n_cal={len(cal_recs)}  n_test={len(test_recs)}")
    print(f"policy cost vector cv = " + ", ".join(f"{a}={cv[i]:.5f}" for i, a in enumerate(ACTIONS)))

    lams = enumerate_breakpoints(P_cal, cv)
    print(f"calib breakpoints: {len(lams)}")
    if not lams:
        lams = [0.5]

    # Diagnostic: min achievable fail rate / cost over λ-sweep on CALIB
    sweep_diag_fail = []
    sweep_diag_cost = []
    for lam in [0.0] + list(lams):
        idx = policy(P_cal, cv, lam)
        sweep_diag_fail.append(float((1 - O_cal[np.arange(len(O_cal)), idx]).mean()))
        sweep_diag_cost.append(float((proceed_cal + recov_cal[np.arange(len(O_cal)), idx]).mean()))
    min_cal_fail = min(sweep_diag_fail)
    min_cal_cost = min(sweep_diag_cost)
    print(f"\nDiagnostic on CALIB over λ-sweep:")
    print(f"  min achievable fail rate = {min_cal_fail:.4f}  → α MUST be > {min_cal_fail + hoeffding_eps(len(O_cal), args.delta):.4f} (with ε)")
    print(f"  min achievable mean cost = {min_cal_cost:.5f}")

    out = {
        "cost_mode": args.cost_mode,
        "n_cal": len(cal_recs),
        "n_test": len(test_recs),
        "delta": args.delta,
        "cv": {a: float(cv[i]) for i, a in enumerate(ACTIONS)},
        "min_cal_fail": min_cal_fail,
        "min_cal_cost": min_cal_cost,
        "eps_hoeffding_fail": hoeffding_eps(len(O_cal), args.delta),
        "baselines_test": baselines(P_test, Y_test, O_test, recov_test, proceed_test),
        "mode_A": [],
        "mode_B": [],
    }

    print("\n=== Baselines on test ===")
    for b in out["baselines_test"]:
        print(f"  {b['name']:18s}  cost={b['mean_cost']:10.5f}  fail={b['fail_rate']:.4f}  "
              f"solve={b['solve_rate']:.4f}  dec_acc={b['dec_acc']:.4f}")

    # ----- Mode A: cost-budget -----
    if args.mode in ("A", "both"):
        if args.cost_mode == "api_call":
            C_sweep = [float(x) for x in args.C_sweep.split(",")]
        else:
            C_sweep = [float(x) for x in args.C_sweep_token.split(",")]
        print("\n=== Mode A: cost-budget — pick smallest λ s.t. E[cost] ≤ C ===")
        print(f"{'C':>10s} {'ε':>8s} {'cal_target':>11s} {'λ̂':>10s} {'cal_cost':>10s} "
              f"{'test_cost':>10s} {'test_fail':>10s} {'solve':>7s} {'held?':>6s}")
        for C in C_sweep:
            cal = calibrate_mode_A(P_cal, recov_cal, proceed_cal, cv, lams, C, args.delta)
            if not cal["feasible"]:
                print(f"  {C:10.5f}  (infeasible — even λ=∞ violates target {cal['target_after_slack']:.5f})")
                out["mode_A"].append({"C": C, "feasible": False, "eps": cal["eps"]})
                continue
            test = eval_lam_on_test(P_test, Y_test, O_test, recov_test, proceed_test, cv, cal["lam_hat"])
            held = test["mean_cost"] <= C
            entry = {**cal, "test": test, "guarantee_held": held}
            out["mode_A"].append(entry)
            print(f"  {C:10.5f} {cal['eps']:8.4f} {cal['target_after_slack']:11.5f} "
                  f"{cal['lam_hat']:10.4g} {cal['cal_mean_cost']:10.5f} "
                  f"{test['mean_cost']:10.5f} {test['fail_rate']:10.4f} {test['solve_rate']:7.4f} "
                  f"{'✓' if held else '✗':>6s}")

    # ----- Mode B: fail-budget -----
    if args.mode in ("B", "both"):
        alphas = [float(x) for x in args.alpha_sweep.split(",")]
        print("\n=== Mode B: fail-budget — pick largest λ s.t. P[fail] ≤ α ===")
        print(f"  (i.e. P(test acc ≥ 1-α) ≥ 1-δ with δ={args.delta})")
        print(f"{'α':>8s} {'ε':>8s} {'cal_target':>11s} {'λ̂':>10s} {'cal_fail':>9s} "
              f"{'test_fail':>10s} {'test_acc':>9s} {'test_cost':>10s} {'held?':>6s}")
        for alpha in alphas:
            cal = calibrate_mode_B(P_cal, O_cal, cv, lams, alpha, args.delta)
            if not cal["feasible"]:
                print(f"  {alpha:8.3f}  (infeasible — no λ satisfies target {cal['target_after_slack']:.4f})")
                out["mode_B"].append({"alpha": alpha, "feasible": False, "eps": cal["eps"]})
                continue
            test = eval_lam_on_test(P_test, Y_test, O_test, recov_test, proceed_test, cv, cal["lam_hat"])
            held = test["fail_rate"] <= alpha
            entry = {**cal, "test": test, "guarantee_held": held}
            out["mode_B"].append(entry)
            print(f"  {alpha:8.3f} {cal['eps']:8.4f} {cal['target_after_slack']:11.4f} "
                  f"{cal['lam_hat']:10.4g} {cal['cal_fail_rate']:9.4f} "
                  f"{test['fail_rate']:10.4f} {test['solve_rate']:9.4f} {test['mean_cost']:10.5f} "
                  f"{'✓' if held else '✗':>6s}")

    # save
    suffix2 = "_4cls" if args.num_classes == 4 else ""
    out_path = args.out or f"/mnt/bn/ecom-govern-models/qijiahe/conformal/results/crc_v3{suffix2}_{args.cost_mode}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
