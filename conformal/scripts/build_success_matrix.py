"""Build per-(problem, action) success matrix O[i, a] in {0,1}.

For each problem in calib_preds_with_pid + test_preds_with_pid, look up the
matching rollout record by (dataset, problem_id) and read summary.successful_actions
to get per-action success {reflect, replan, escalate}.

We only track recovery actions — proceed is filtered upstream (calib/test only
contain proceed-fail problems).

Output: success_matrix.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROLLOUT_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/cloudide/rollout")
OUT_PATH = Path("/mnt/bn/ecom-govern-models/qijiahe/conformal/data/success_matrix.json")

ACTIONS = ("reflect", "replan", "escalate")


def main():
    # build (dataset, problem_id) -> successful_actions set
    succ = {}
    for jpath in sorted(ROLLOUT_DIR.glob("*.jsonl")):
        if jpath.name.endswith(".pretty.txt"):
            continue
        dataset = jpath.stem
        with jpath.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "summary" not in rec:
                    continue
                key = f"{dataset}::{rec['problem_id']}"
                actions = set(rec["summary"].get("successful_actions", []))
                succ[key] = actions
    print(f"loaded successful_actions for {len(succ)} (dataset, problem_id) pairs")

    # Filter to calib + test problems only, build matrix
    matrix = {}
    n_no_succ = 0
    for split in ("calib", "test"):
        path = Path(f"/mnt/bn/ecom-govern-models/qijiahe/conformal/data/{split}_preds_with_pid.jsonl")
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['dataset']}::{r['problem_id']}"
                if key not in succ:
                    n_no_succ += 1
                    continue
                actions = succ[key]
                matrix[key] = {a: int(a in actions) for a in ACTIONS}
                # also include the proceed indicator (will be 0 in calib/test since proceed-fail filter, but defensive)
                matrix[key]["proceed"] = int("proceed" in actions)
                # `unsolvable` outcome: 1 iff NO action passed (oracle_unsolvable)
                # In 3cls-derived calib/test this is always 0 (filtered out upstream),
                # but the 4cls policy may still propose it — having it in the matrix
                # lets the 4cls eval score those decisions correctly.
                matrix[key]["unsolvable"] = int(len(actions) == 0)
    if n_no_succ:
        print(f"  warning: {n_no_succ} calib/test rows had no matching rollout record")

    # Summary stats per split
    print()
    for split in ("calib", "test"):
        path = Path(f"/mnt/bn/ecom-govern-models/qijiahe/conformal/data/{split}_preds_with_pid.jsonl")
        keys = []
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                keys.append(f"{r['dataset']}::{r['problem_id']}")
        sums = {a: 0 for a in ACTIONS}
        n = 0
        for k in keys:
            if k in matrix:
                n += 1
                for a in ACTIONS:
                    sums[a] += matrix[k][a]
        print(f"  {split}  n={n}")
        for a in ACTIONS:
            print(f"    P({a} passes) = {sums[a]/n:.3f}  ({sums[a]}/{n})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fout:
        json.dump(matrix, fout, indent=2)
    print(f"\nsaved -> {OUT_PATH} ({len(matrix)} entries)")


if __name__ == "__main__":
    main()
