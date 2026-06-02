"""Side-by-side comparison of CRC sweeps across all available classifiers.

Reads:
  results/eval.json           (router CRC sweep)
  results/eval_nano.json      (GPT-5.4-nano CRC sweep)
  results/eval_qwenbase.json  (Qwen3.5-4B base, no LoRA)
  results/eval_mini.json      (GPT-5.4-mini CRC sweep, optional)
Writes:
  results/comparison.md       (markdown table)
"""
import json
from pathlib import Path


CLASSIFIERS = [
    ("router  ", "results/eval.json"),
    ("qwenbase", "results/eval_qwenbase.json"),
    ("nano    ", "results/eval_nano.json"),
    ("mini    ", "results/eval_mini.json"),
]


def fmt_row(name, row, n_test):
    cd = row["chosen_dist"]
    return (
        f"| {name} | {row['alpha']:.2f} | {row['tau']:.4f} | "
        f"{row['coverage']:.3f} | {row['empirical_risk']:.3f} | "
        f"{row['decision_accuracy']:.4f} | {row['mean_set_size']:.3f} | "
        f"{row['mean_cost']:.3f} | {cd['reflect']/n_test*100:.1f}% | "
        f"{cd['replan']/n_test*100:.1f}% | {cd['escalate']/n_test*100:.1f}% | "
        f"{row['empty_set_fallbacks']} |"
    )


def main():
    base = Path("/mnt/bn/ecom-govern-models/qijiahe/conformal")
    loaded = []  # list of (name, eval_dict)
    for name, path in CLASSIFIERS:
        p = base / path
        if not p.exists():
            print(f"skip {name.strip()} (no {path})")
            continue
        with open(p) as f:
            loaded.append((name, json.load(f)))

    lines = []
    lines.append("# Router (SFT Qwen3.5-4B) vs Qwen-base vs nano vs mini — CRC comparison")
    lines.append("")
    lines.append("## Baselines (argmax-only, no CRC)")
    lines.append("")
    lines.append("| classifier | baseline | dec_acc | mean_cost |")
    lines.append("|---|---|---|---|")
    for name, evald in loaded:
        for b in evald["baselines"]:
            lines.append(f"| {name} | {b['name']} | {b['decision_accuracy']:.4f} | {b['mean_cost']:.3f} |")

    lines.append("")
    lines.append("## CRC alpha sweep")
    lines.append("")
    lines.append("| classifier | α | τ | coverage | risk | dec_acc | |set| | cost | refl% | repl% | esc% | empty |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    all_alphas = sorted({r["alpha"] for _, evald in loaded for r in evald["crc_sweep"]})
    for a in all_alphas:
        for name, evald in loaded:
            n_test = evald["crc_sweep"][0]["n_test"]
            row = next((r for r in evald["crc_sweep"] if abs(r["alpha"] - a) < 1e-6), None)
            if row:
                lines.append(fmt_row(name, row, n_test))

    out = base / "results/comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
