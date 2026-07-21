"""Run CRC Mode A (E[cost] <= C) on holdout with per-example USD costs.

Same predefined calib/test split design as crc_on_holdout.py, but cost(a) is
no longer a fixed {reflect:2, replan:2, escalate:13} constant -- each example
has its own real per-action USD cost (from attach_usd_costs.py output), so
the candidate lambda breakpoints and the cost lookup are computed per example.

Usage:
    python scripts/crc_on_holdout_usd.py \
        --calib_eval /path/to/calib_eval_usd.json \
        --test_eval  /path/to/test_eval_usd.json \
        --out        /path/to/results_usd.json \
        --budgets 0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012
"""
import argparse
import json
import math
from pathlib import Path

CANDIDATES = ["reflect", "replan", "escalate"]


def policy(probs, costs, lam):
    return max(CANDIDATES, key=lambda a: probs[a] - lam * costs[a])


def hoeffding_eps(n, delta, lo, hi):
    return (hi - lo) * math.sqrt(math.log(1 / delta) / (2 * n))


def usable(examples):
    return [ex for ex in examples if ex.get("costs_usd") is not None]


def eval_split(examples, lambdas):
    results = {}
    for lam in lambdas:
        costs_realized, fails = [], []
        for ex in examples:
            pred = policy(ex["probs"], ex["costs_usd"], lam)
            costs_realized.append(ex["costs_usd"][pred])
            sa = ex.get("successful_actions")
            if sa is not None:
                fails.append(0 if pred in sa else 1)
        results[lam] = {
            "cost": sum(costs_realized) / len(costs_realized) if costs_realized else 0,
            "fail": sum(fails) / len(fails) if fails else None,
            "solve": (1 - sum(fails) / len(fails)) if fails else None,
            "pred_dist": {
                a: sum(1 for ex in examples if policy(ex["probs"], ex["costs_usd"], lam) == a)
                for a in CANDIDATES
            },
        }
    return results


def build_lambdas(examples):
    lambdas = {0.0}
    for ex in examples:
        p, c = ex["probs"], ex["costs_usd"]
        for a1 in CANDIDATES:
            for a2 in CANDIDATES:
                if a1 == a2:
                    continue
                dc = c[a1] - c[a2]
                if dc != 0:
                    lam = (p[a1] - p[a2]) / dc
                    if lam > 0:
                        lambdas.add(lam)
    return sorted(lambdas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib_eval", required=True)
    p.add_argument("--test_eval", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--budgets", default="0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012")
    args = p.parse_args()

    calib_exs = usable(json.load(open(args.calib_eval))["examples"])
    test_exs = usable(json.load(open(args.test_eval))["examples"])
    n_cal, n_test = len(calib_exs), len(test_exs)

    all_costs = [v for ex in calib_exs + test_exs for v in ex["costs_usd"].values()]
    lo, hi = min(all_costs), max(all_costs)
    eps_cost = hoeffding_eps(n_cal, args.delta, lo, hi)
    eps_fail = hoeffding_eps(n_cal, args.delta, 0, 1)
    print(f"n_calib={n_cal} n_test={n_test} cost_range=[{lo:.5f},{hi:.5f}] "
          f"eps_cost={eps_cost:.6f} eps_fail={eps_fail:.4f}")

    lambdas = build_lambdas(calib_exs)
    print(f"breakpoints: {len(lambdas)}")

    calib_vals = eval_split(calib_exs, lambdas)
    test_vals = eval_split(test_exs, lambdas)

    argmax_calib = calib_vals[0.0]
    argmax_test = test_vals[0.0]
    print(f"argmax calib: solve={argmax_calib['solve']:.4f} cost=${argmax_calib['cost']:.5f}")
    print(f"argmax test:  solve={argmax_test['solve']:.4f} cost=${argmax_test['cost']:.5f}")

    budgets = [float(x) for x in args.budgets.split(",")]
    mode_a = []
    print("\nMode A (E[cost] <= C):")
    for C in budgets:
        target = C - eps_cost
        feasible = [(lam, calib_vals[lam]) for lam in lambdas if calib_vals[lam]["cost"] <= target]
        if feasible:
            best_lam, best_cal = min(feasible, key=lambda x: x[0])
            t = test_vals[best_lam]
            held = t["cost"] <= C
            entry = {
                "C": C, "target": round(target, 6), "lambda": best_lam,
                "calib_cost": best_cal["cost"], "calib_solve": best_cal["solve"],
                "test_cost": t["cost"], "test_solve": t["solve"],
                "guarantee_held": held, "feasible": True,
            }
            print(f"  C=${C:.4f}  lam={best_lam:.2f}  test_cost=${t['cost']:.5f}  "
                  f"test_solve={t['solve']:.4f}  {'OK' if held else 'X'}")
        else:
            entry = {"C": C, "feasible": False}
            print(f"  C=${C:.4f}  infeasible")
        mode_a.append(entry)

    out = {
        "n_calib": n_cal, "n_test": n_test, "delta": args.delta,
        "cost_lo": lo, "cost_hi": hi, "eps_cost": eps_cost, "eps_fail": eps_fail,
        "argmax_calib": argmax_calib, "argmax_test": argmax_test,
        "mode_a": mode_a,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
