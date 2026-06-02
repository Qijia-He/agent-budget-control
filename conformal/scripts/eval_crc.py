"""
Evaluate the calibrated CRC on the test split.

For each alpha (with its calibrated tau):
  prediction_set(x) = {a : p_a(x) >= 1 - tau}
  chosen_action = cheapest action in prediction_set(x), tie-broken by COST order
  if true label in prediction_set: covered (counts toward 1 - empirical_risk)
  if chosen_action == true_cheapest_label: "correct decision"
  cost := COST[chosen_action]  (if set empty, fall back to "escalate")

We report:
  - empirical risk (1 - coverage)
  - average cost per example
  - average prediction-set size
  - argmax-only baseline (no CRC, no prediction set)
  - always-escalate baseline
  - always-reflect baseline (cheapest)
"""
import json
import argparse
import numpy as np
from pathlib import Path


COST = {"reflect": 2.0, "replan": 2.0, "escalate": 13.0}  # nano-units
CHEAPER_ORDER = ["reflect", "replan", "escalate"]  # tie-break preference (cheaper first)


def cheapest_in_set(prediction_set):
    """Pick the cheapest action in the set, tie-break by CHEAPER_ORDER."""
    if not prediction_set:
        # empty set: fall back to safest (escalate)
        return "escalate", True
    by_cost = sorted(prediction_set, key=lambda a: (COST[a], CHEAPER_ORDER.index(a)))
    return by_cost[0], False


def evaluate_alpha(records, tau):
    n = len(records)
    n_covered = 0
    n_correct_decision = 0
    set_sizes = []
    costs = []
    chosen_dist = {"reflect": 0, "replan": 0, "escalate": 0}
    fallback_count = 0
    for rec in records:
        probs = rec["probs"]
        true_label = rec["true_label"]
        # prediction set: labels with p >= 1 - tau
        thresh = 1.0 - tau
        pset = [a for a, p in probs.items() if p >= thresh]
        set_sizes.append(len(pset))
        if true_label in pset:
            n_covered += 1
        chosen, fallback = cheapest_in_set(pset)
        if fallback:
            fallback_count += 1
        chosen_dist[chosen] = chosen_dist.get(chosen, 0) + 1
        costs.append(COST[chosen])
        if chosen == true_label:
            n_correct_decision += 1
    return {
        "n": n,
        "coverage": n_covered / n,
        "empirical_risk": 1.0 - n_covered / n,
        "decision_accuracy": n_correct_decision / n,
        "mean_set_size": float(np.mean(set_sizes)),
        "set_size_dist": {str(k): set_sizes.count(k) for k in range(0, 4)},
        "mean_cost": float(np.mean(costs)),
        "p50_cost": float(np.median(costs)),
        "chosen_dist": chosen_dist,
        "empty_set_fallbacks": fallback_count,
    }


def evaluate_baseline_argmax(records):
    n = len(records)
    correct = 0
    costs = []
    for rec in records:
        argmax_action = max(rec["probs"], key=rec["probs"].get)
        if argmax_action == rec["true_label"]:
            correct += 1
        costs.append(COST[argmax_action])
    return {"name": "argmax", "n": n, "decision_accuracy": correct / n,
            "mean_cost": float(np.mean(costs))}


def evaluate_baseline_fixed(records, action):
    n = len(records)
    correct = sum(1 for r in records if r["true_label"] == action)
    return {"name": f"always_{action}", "n": n, "decision_accuracy": correct / n,
            "mean_cost": COST[action]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_preds", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/test_preds.jsonl")
    p.add_argument("--tau_table", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/tau_table.json")
    p.add_argument("--output", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/results/eval.json")
    args = p.parse_args()

    records = []
    with open(args.test_preds) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"loaded {len(records)} test records")
    label_dist = {}
    for r in records:
        label_dist[r["true_label"]] = label_dist.get(r["true_label"], 0) + 1
    print(f"test label distribution: {label_dist}")

    with open(args.tau_table) as f:
        tau_table = json.load(f)

    result = {
        "baselines": [
            evaluate_baseline_argmax(records),
            evaluate_baseline_fixed(records, "reflect"),
            evaluate_baseline_fixed(records, "replan"),
            evaluate_baseline_fixed(records, "escalate"),
        ],
        "crc_sweep": [],
    }

    print()
    print("=== baselines ===")
    for b in result["baselines"]:
        print(f"  {b['name']:18s} dec_acc={b['decision_accuracy']:.4f}  mean_cost={b['mean_cost']:.3f}")

    print()
    print("=== CRC alpha sweep ===")
    print(f"{'alpha':>6s} {'tau':>8s} {'cov':>6s} {'risk':>6s} {'dec_acc':>8s} "
          f"{'|set|':>6s} {'cost':>6s} {'reflect':>8s} {'replan':>8s} {'escalate':>9s} {'empty':>6s}")
    for key, info in tau_table["alphas"].items():
        alpha = info["alpha"]
        tau = info["tau"]
        metrics = evaluate_alpha(records, tau)
        n = metrics["n"]
        entry = {
            "alpha": alpha,
            "tau": tau,
            "n_calib": tau_table["n_calib"],
            "n_test": n,
            **metrics,
        }
        result["crc_sweep"].append(entry)
        cd = metrics["chosen_dist"]
        print(f"{alpha:6.2f} {tau:8.4f} {metrics['coverage']:6.3f} {metrics['empirical_risk']:6.3f} "
              f"{metrics['decision_accuracy']:8.4f} {metrics['mean_set_size']:6.3f} "
              f"{metrics['mean_cost']:6.3f} {cd['reflect']/n*100:7.1f}% {cd['replan']/n*100:7.1f}% "
              f"{cd['escalate']/n*100:8.1f}% {metrics['empty_set_fallbacks']:6d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        json.dump(result, fout, indent=2)
    print()
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()
