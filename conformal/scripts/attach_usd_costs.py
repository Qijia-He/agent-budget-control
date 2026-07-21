"""Attach per-problem USD action costs to an eval_router_benchmark.py output file.

eval_router_benchmark.py writes examples in the same order as the input
benchmark JSON but drops the `_dataset` / `_problem_id` fields. Recover them
by re-zipping against the original benchmark file (same order, same length),
then look up real per-(dataset, problem_id) token costs from
conformal/data/action_costs_usd.json. Falls back to the dataset-level mean
for the ~1-3% of problems not covered by the cost table.

Use --refresh_success_labels after benchmark SA has been updated, e.g. by
run_benchmark_exhaustive.py, so stale eval JSON labels are replaced by the
verified benchmark labels before CRC is recomputed.

Usage:
    python conformal/scripts/attach_usd_costs.py \
        --eval_json   datasets/eval_results/holdout_3cls_v4/holdout_3cls_test_eval.json \
        --bench_json  datasets/benchmarks/holdout_3cls_test.json \
        --costs_json  conformal/data/action_costs_usd.json \
        --out         datasets/eval_results/holdout_3cls_v4/holdout_3cls_test_eval_usd.json
"""
import argparse
import json
from pathlib import Path

CANDIDATES = ["reflect", "replan", "escalate"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", required=True)
    p.add_argument("--bench_json", required=True)
    p.add_argument("--costs_json", default="conformal/data/action_costs_usd.json")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--refresh_success_labels",
        action="store_true",
        help="copy successful_actions/oracle flags from bench_json into eval examples",
    )
    args = p.parse_args()

    eval_data = json.load(open(args.eval_json))
    bench_data = json.load(open(args.bench_json))
    costs = json.load(open(args.costs_json))
    per_problem = costs["per_problem"]
    dataset_means = costs["dataset_means_usd"]

    assert len(eval_data["examples"]) == len(bench_data), (
        f"length mismatch: eval has {len(eval_data['examples'])}, "
        f"bench has {len(bench_data)} -- not the same file/order"
    )

    n_exact, n_imputed, n_missing = 0, 0, 0
    for ex, bex in zip(eval_data["examples"], bench_data):
        dataset = bex.get("_dataset")
        problem_id = bex.get("_problem_id")
        key = f"{dataset}::{problem_id}"
        row = per_problem.get(key)
        if row is not None:
            usd = {a: row[f"{a}_usd"] for a in CANDIDATES}
            n_exact += 1
        elif dataset in dataset_means:
            usd = {a: dataset_means[dataset][a] for a in CANDIDATES}
            n_imputed += 1
        else:
            usd = None
            n_missing += 1
        ex["_dataset"] = dataset
        ex["_problem_id"] = problem_id
        ex["costs_usd"] = usd
        if args.refresh_success_labels:
            for key in [
                "successful_actions",
                "oracle_unsolvable",
                "_escalate_tested",
                "_escalate_ctx_verdict",
            ]:
                if key in bex:
                    ex[key] = bex[key]

    eval_data["usd_cost_coverage"] = {
        "n_exact": n_exact, "n_dataset_mean_imputed": n_imputed, "n_missing": n_missing,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_data, open(args.out, "w"), indent=2)
    print(f"{args.out}: exact={n_exact} imputed={n_imputed} missing={n_missing}")


if __name__ == "__main__":
    main()
