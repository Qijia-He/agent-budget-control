"""
Split the 3cls SFT data into train / calibration / test (80/10/10, seeded).

Note on the holdout caveat:
  the router was trained on a LLaMA-Factory random 90/10 train/eval split with
  its own seed. The eval portion (159 examples) is the only data the router
  has never seen. We don't know LLaMA-Factory's seed, so we cannot exactly
  reproduce its eval set. For "链路跑通" (pipeline-validation) we accept that
  our calib+test split may partially overlap with the router's SFT training
  set — the router will be overconfident on overlapped examples, which biases
  CRC's empirical risk *downward*. Flag this in FINDINGS.md.

  For paper-quality CRC, we'd need either:
    (a) a fresh held-out dataset, or
    (b) reproducing LLaMA-Factory's exact eval split and partitioning it.
"""
import json
import argparse
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/mnt/bn/ecom-govern-models/qijiahe/datasets/router_no_reason_v1_3cls.json")
    p.add_argument("--out_dir", default="/mnt/bn/ecom-govern-models/qijiahe/conformal/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", default="0.8,0.1,0.1", help="train,calib,test fractions")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)
    n = len(data)
    print(f"loaded {n} examples")

    ratios = [float(x) for x in args.ratios.split(",")]
    assert abs(sum(ratios) - 1.0) < 1e-6
    n_train = int(n * ratios[0])
    n_calib = int(n * ratios[1])
    # test gets the remainder so counts sum to n exactly
    n_test = n - n_train - n_calib

    rng = random.Random(args.seed)
    idx = list(range(n))
    rng.shuffle(idx)

    splits = {
        "train": [data[i] for i in idx[:n_train]],
        "calib": [data[i] for i in idx[n_train:n_train + n_calib]],
        "test":  [data[i] for i in idx[n_train + n_calib:]],
    }

    from collections import Counter
    for name, items in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        labels = Counter(ex["output"] for ex in items)
        print(f"  {name}: n={len(items)} labels={dict(labels)} -> {path}")


if __name__ == "__main__":
    main()
