"""Pretty-print rollout records from a jsonl file.

Usage:
    # all records of a file
    python scripts/view_rollouts.py outputs/rollouts/v55/bcb.jsonl

    # specific problem (substring match on problem_id)
    python scripts/view_rollouts.py outputs/rollouts/v55/bcb.jsonl --filter "BigCodeBench/0"

    # only failed / diagnose-present records
    python scripts/view_rollouts.py outputs/rollouts/v55/bcb.jsonl --only_failed

    # write pretty version to a file
    python scripts/view_rollouts.py outputs/rollouts/v55/bcb.jsonl --out bcb.pretty.txt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_rollouts_opt import _format_pretty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_path")
    p.add_argument("--filter", default=None,
                   help="substring match on problem_id; only matching records shown")
    p.add_argument("--only_failed", action="store_true",
                   help="only records where proceed failed (i.e. have diagnose)")
    p.add_argument("--out", default=None,
                   help="write to this file (default: stdout)")
    args = p.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"file not found: {path}")
        sys.exit(1)

    out = open(args.out, "w") if args.out else sys.stdout
    n_shown = 0
    n_total = 0
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                n_total += 1
                try:
                    rec = json.loads(line)
                except Exception as e:
                    print(f"[parse error] line {n_total}: {e}", file=sys.stderr)
                    continue
                if args.filter and args.filter not in rec.get("problem_id", ""):
                    continue
                if args.only_failed and not rec.get("diagnose"):
                    continue
                out.write(_format_pretty(rec))
                n_shown += 1
    finally:
        if args.out:
            out.close()
    print(f"\nshown {n_shown}/{n_total} records", file=sys.stderr)


if __name__ == "__main__":
    main()
