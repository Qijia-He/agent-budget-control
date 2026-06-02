"""
Split-conformal calibration over the calibration jsonl.

Nonconformity score: s_i = 1 - p_i[y_i^true]
  (large s_i = router gave the true label low probability = nonconforming)

For a target risk alpha:
  tau = quantile_(1-alpha) of the calibration scores, with conformal
        finite-sample correction: take the ceil((n+1)(1-alpha))/n quantile.

Deployment-time prediction set:
  C(x) = { a : 1 - p_a(x) <= tau }  =  { a : p_a(x) >= 1 - tau }

We compute tau for a sweep of alpha values and save them all to one file.
"""
import json
import argparse
import math
import numpy as np
from pathlib import Path


def conformal_quantile(scores, alpha):
    """Split-conformal quantile with finite-sample correction.

    Returns tau such that on the calibration set
        P(score <= tau) >= ceil((n+1)(1-alpha)) / n
    so test-time coverage P(s_test <= tau) >= 1 - alpha (Vovk style).
    """
    n = len(scores)
    # rank from smallest. We want the index k = ceil((n+1)(1-alpha)) - 1 (0-indexed)
    k = math.ceil((n + 1) * (1 - alpha)) - 1
    if k >= n:
        # alpha so small that (n+1)(1-alpha) > n -> impossible to guarantee at this n
        return float("inf")  # equivalent to "no scoreis nonconforming enough", prediction set = all labels
    if k < 0:
        return float("-inf")
    return float(np.partition(scores, k)[k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/calib_preds.jsonl")
    p.add_argument("--output", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data/tau_table.json")
    p.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
                   help="comma-separated alpha sweep")
    args = p.parse_args()

    scores = []
    n_records = 0
    with open(args.calib) as f:
        for line in f:
            rec = json.loads(line)
            n_records += 1
            p_true = rec["probs"][rec["true_label"]]
            scores.append(1.0 - p_true)
    scores = np.array(scores)
    print(f"loaded {n_records} calibration records")
    print(f"score (1 - p_true) stats: mean={scores.mean():.4f} std={scores.std():.4f} "
          f"min={scores.min():.4f} max={scores.max():.4f}")

    alphas = [float(x) for x in args.alphas.split(",")]
    table = {"n_calib": len(scores), "alphas": {}}
    for alpha in alphas:
        tau = conformal_quantile(scores, alpha)
        table["alphas"][f"{alpha:.4f}"] = {"alpha": alpha, "tau": tau}
        print(f"  alpha={alpha:.2f} -> tau={tau:.4f}  "
              f"(prediction set includes labels with p>=1-tau={1-tau:.4f})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        json.dump(table, fout, indent=2)
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()
