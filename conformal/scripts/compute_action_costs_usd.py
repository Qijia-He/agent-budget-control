"""Build per-(problem, action) USD cost table from Phase A rollouts.

For each proceed-fail problem in rollout/*.jsonl, extract per-call cost_usd:
  - proceed_usd:  always available (proceed was attempted)
  - reflect_usd:  always available (short_circuit_runner runs reflect+replan
                  in tandem after proceed-fail)
  - replan_usd:   always available (same reason)
  - escalate_usd: only present when both reflect AND replan failed. For
                  problems where escalate was short-circuited away, impute
                  the dataset-mean of observed escalate_usd. Flag count.

The CRC λ-policy uses these per-action *recovery* costs (NOT including the
sunk proceed cost). Final per-problem cost reported by eval = proceed + chosen.

Output: action_costs_usd.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROLLOUT_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/cloudide/rollout")
OUT_PATH = Path("/mnt/bn/ecom-govern-models/qijiahe/conformal/data/action_costs_usd.json")


def main():
    per_problem = {}
    by_dataset_observed = defaultdict(lambda: defaultdict(list))
    n_seen = 0
    n_proceed_pass = 0
    n_escalate_observed = 0
    n_escalate_missing = 0

    for jpath in sorted(ROLLOUT_DIR.glob("*.jsonl")):
        if jpath.name.endswith(".pretty.txt"):
            continue
        dataset = jpath.stem
        with jpath.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "calls" not in rec or not rec["calls"]:
                    continue
                n_seen += 1
                calls = rec["calls"]
                if calls[0]["verdict"] == "pass":
                    n_proceed_pass += 1
                    # still record proceed cost for reference, but skip the rest
                    continue

                by_action = {c["action"]: float(c.get("cost_usd") or 0.0)
                             for c in calls}
                # Sanity: proceed-fail records should always have proceed + reflect + replan
                proceed_usd = by_action.get("proceed", 0.0)
                reflect_usd = by_action.get("reflect")
                replan_usd  = by_action.get("replan")
                escalate_usd = by_action.get("escalate")  # may be None

                pid = rec["problem_id"]
                # if duplicate problem_id across datasets, prefix with dataset
                key = f"{dataset}::{pid}"
                per_problem[key] = {
                    "dataset": dataset,
                    "problem_id": pid,
                    "proceed_usd": proceed_usd,
                    "reflect_usd": reflect_usd,
                    "replan_usd":  replan_usd,
                    "escalate_usd": escalate_usd,  # may be None
                    "unsolvable_usd": 0.0,         # no API call = no $ cost
                }

                if reflect_usd is not None:
                    by_dataset_observed[dataset]["reflect"].append(reflect_usd)
                if replan_usd is not None:
                    by_dataset_observed[dataset]["replan"].append(replan_usd)
                if escalate_usd is not None:
                    n_escalate_observed += 1
                    by_dataset_observed[dataset]["escalate"].append(escalate_usd)
                else:
                    n_escalate_missing += 1
                by_dataset_observed[dataset]["proceed"].append(proceed_usd)

    # Compute dataset means for imputation
    dataset_means = {}
    for ds, by_act in by_dataset_observed.items():
        dataset_means[ds] = {a: mean(v) for a, v in by_act.items() if v}

    # Apply imputation
    n_imputed = 0
    for key, rec in per_problem.items():
        ds = rec["dataset"]
        for a in ("reflect", "replan", "escalate"):
            if rec[f"{a}_usd"] is None:
                fallback = dataset_means.get(ds, {}).get(a)
                if fallback is None:
                    fallback = 0.0  # last resort
                rec[f"{a}_usd"] = fallback
                rec.setdefault("imputed", []).append(a)
                n_imputed += 1

    print(f"[scan] n_seen={n_seen} proceed_pass={n_proceed_pass} "
          f"proceed_fail={len(per_problem)}")
    print(f"[scan] escalate observed={n_escalate_observed} missing={n_escalate_missing}")
    print(f"[impute] action-slots imputed with dataset-mean: {n_imputed}")
    print()
    print("dataset_means_usd ($):")
    for ds, m in dataset_means.items():
        print(f"  {ds:30s}  " + "  ".join(
            f"{a}={m.get(a, float('nan')):.5f}" for a in ("proceed","reflect","replan","escalate")
        ))

    out = {
        "dataset_means_usd": dataset_means,
        "per_problem": per_problem,
        "_counts": {
            "n_seen": n_seen,
            "n_proceed_pass": n_proceed_pass,
            "n_proceed_fail": len(per_problem),
            "escalate_observed": n_escalate_observed,
            "escalate_missing_imputed": n_escalate_missing,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fout:
        json.dump(out, fout, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
