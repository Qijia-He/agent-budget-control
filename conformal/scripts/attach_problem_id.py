"""Attach problem_id to calib_preds.jsonl / test_preds.jsonl.

The previous data_split.py shuffled router_no_reason_v1_3cls.json by random seed
and lost the (dataset, problem_id) anchor for each example. To compute per-problem
token cost from rollouts, we need to recover problem_id per row.

Strategy:
  1. Walk rollout/*.jsonl. For each proceed-fail record, rebuild the SFT
     `input` string with the same logic as build_sft_router_v1.build_input.
  2. Hash the input text → store in dict {input_hash: (dataset, problem_id)}.
  3. For each row in calib.jsonl / test.jsonl, hash its input and look up.
     Emit calib_preds_with_pid.jsonl / test_preds_with_pid.jsonl (preds + pid).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROLLOUT_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/cloudide/rollout")
DATA_DIR = Path("/mnt/bn/ecom-govern-models/qijiahe/conformal/data")

PROMPT_CHARS = 6000
STDERR_CAP_FAIL = 800
STDERR_CAP_OTHER = 400


def trim_stderr(verdict: str, stderr: str) -> str:
    s = (stderr or "").strip()
    if not s:
        return ""
    cap = STDERR_CAP_FAIL if verdict == "fail" else STDERR_CAP_OTHER
    if len(s) > cap:
        s = s[:cap] + "\n... [truncated]"
    return s


def build_input(rec) -> str:
    prompt = rec["problem_prompt"]
    if len(prompt) > PROMPT_CHARS:
        prompt = prompt[:PROMPT_CHARS] + "\n... [problem truncated]"
    c0 = rec["calls"][0]
    verdict = c0["verdict"]
    stderr = trim_stderr(verdict, c0.get("stderr") or "")
    parts = ["Problem:", prompt, "", f"Initial attempt verdict: {verdict}"]
    if stderr:
        parts.append("Initial attempt stderr:")
        parts.append(stderr)
    return "\n".join(parts)


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_index():
    idx = {}
    n_seen = 0
    n_proceed_pass = 0
    n_collision = 0
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
                if rec["calls"][0]["verdict"] == "pass":
                    n_proceed_pass += 1
                    continue
                inp = build_input(rec)
                key = h(inp)
                if key in idx and idx[key] != (dataset, rec["problem_id"]):
                    n_collision += 1
                idx[key] = (dataset, rec["problem_id"])
    print(f"[index] n_seen={n_seen} proceed_pass_skipped={n_proceed_pass} "
          f"unique_keys={len(idx)} collisions={n_collision}")
    return idx


def attach(split_name: str, idx: dict, preds_suffix: str = ""):
    """preds_suffix='' loads calib_preds.jsonl; '_4cls' loads calib_preds_4cls.jsonl."""
    src_data = DATA_DIR / f"{split_name}.jsonl"          # has the input field
    src_preds = DATA_DIR / f"{split_name}_preds{preds_suffix}.jsonl"
    out = DATA_DIR / f"{split_name}_preds{preds_suffix}_with_pid.jsonl"

    # zip them by line idx — preds was written 1:1 with split_data
    with src_data.open() as f1, src_preds.open() as f2, out.open("w") as fout:
        n_hit = n_miss = 0
        for line1, line2 in zip(f1, f2):
            row_data = json.loads(line1)
            row_pred = json.loads(line2)
            key = h(row_data["input"])
            hit = idx.get(key)
            if hit is None:
                n_miss += 1
                dataset, pid = None, None
            else:
                n_hit += 1
                dataset, pid = hit
            row_pred["dataset"] = dataset
            row_pred["problem_id"] = pid
            fout.write(json.dumps(row_pred, ensure_ascii=False) + "\n")
    print(f"[{split_name}] hit={n_hit} miss={n_miss}  -> {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--preds_suffix", default="",
                   help="'' for 3cls (calib_preds.jsonl); '_4cls' for 4cls (calib_preds_4cls.jsonl)")
    args = p.parse_args()
    idx = build_index()
    attach("calib", idx, args.preds_suffix)
    attach("test", idx, args.preds_suffix)


if __name__ == "__main__":
    main()
