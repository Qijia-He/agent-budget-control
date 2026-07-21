"""
Generate two figures for §5.2.2 (Recovery Actions Are Not an Ordered Ladder):

Fig A: S(x) category distribution — horizontal stacked bar
Fig B: RACER-analog — for cheap-only / both-work / escalate-only groups,
        compare always-escalate vs. oracle on (solve rate, mean USD cost)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path

REPO = Path("/path/to/agent-budget-control")
OUT  = REPO / "overleaf/figures"

# ── data ──────────────────────────────────────────────────────────────────────
calib = json.load(open(REPO / "datasets/benchmarks/holdout_3cls_calib.json"))
test  = json.load(open(REPO / "datasets/benchmarks/holdout_3cls_test.json"))
examples = calib + test  # n=720

costs_db = json.load(open(REPO / "conformal/data/action_costs_usd.json"))
per_prob  = costs_db["per_prob"] if "per_prob" in costs_db else costs_db.get("per_problem", {})
ds_means  = costs_db["dataset_means_usd"]

def get_cost(ex, action):
    key = f"{ex['_dataset']}::{ex['_problem_id']}"
    if key in per_prob and f"{action}_usd" in per_prob[key]:
        return per_prob[key][f"{action}_usd"]
    ds = ex.get("_dataset", "")
    return ds_means.get(ds, {}).get(action, ds_means["apps"][action])

def sa_category(ex):
    sa = frozenset(ex.get("successful_actions", []))
    labels = {
        frozenset(["escalate"]):                        "escalate\nonly",
        frozenset(["replan"]):                          "replan\nonly",
        frozenset(["reflect"]):                         "reflect\nonly",
        frozenset(["reflect", "replan"]):               "reflect\n+ replan",
        frozenset(["escalate", "replan"]):              "escalate\n+ replan",
        frozenset(["escalate", "reflect"]):             "escalate\n+ reflect",
        frozenset(["escalate", "reflect", "replan"]):   "all three",
    }
    return labels.get(sa, "other")

# ── colour palette (colourblind-friendly) ────────────────────────────────────
C_ESC   = "#E06C6C"   # red-ish  — escalate required
C_CHEAP = "#6CB4E4"   # blue-ish — cheap only
C_BOTH  = "#F0B429"   # amber    — both work
C_GRID  = "#CCCCCC"

# ══════════════════════════════════════════════════════════════════════════════
# Figure A: S(x) category horizontal bar
# ══════════════════════════════════════════════════════════════════════════════
ORDER = [
    "escalate\nonly",
    "escalate\n+ replan",
    "escalate\n+ reflect",
    "all three",
    "reflect\n+ replan",
    "replan\nonly",
    "reflect\nonly",
]
GROUP_COLOUR = {
    "escalate\nonly":     C_ESC,
    "escalate\n+ replan": "#E8938A",
    "escalate\n+ reflect":"#EBB0AB",
    "all three":          "#C97070",
    "reflect\n+ replan":  "#8CC8E8",
    "replan\nonly":       C_CHEAP,
    "reflect\nonly":      "#A8D4F0",
}

cats = [sa_category(ex) for ex in examples]
counts = Counter(cats)
n = len(examples)

fig, ax = plt.subplots(figsize=(7, 3.2))
y = 0
bar_h = 0.55
labels_pct = []
for cat in ORDER:
    cnt  = counts.get(cat, 0)
    pct  = cnt / n * 100
    ax.barh(y, pct, height=bar_h, color=GROUP_COLOUR[cat], edgecolor="white", linewidth=0.5)
    ax.text(pct + 0.4, y, f"{pct:.1f}%  (n={cnt})", va="center", fontsize=8.5)
    y += 1

ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(ORDER, fontsize=9)
ax.set_xlabel("Fraction of holdout examples (%)", fontsize=9)
ax.set_xlim(0, 62)
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(left=False)

# legend: three groups
esc_patch   = mpatches.Patch(color=C_ESC,   label="Escalate required")
cheap_patch = mpatches.Patch(color=C_CHEAP, label="Cheap recovery sufficient")
both_patch  = mpatches.Patch(color=C_BOTH,  label="Both work")
ax.legend(handles=[esc_patch, cheap_patch], loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor=C_GRID)

ax.set_title("Successful-action set $S(x)$ distribution  ($n=720$)", fontsize=10, pad=8)
plt.tight_layout()
out_a = OUT / "fig_sx_distribution.pdf"
plt.savefig(out_a, bbox_inches="tight")
plt.savefig(str(out_a).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
plt.close()
print(f"saved → {out_a}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure B: RACER-analog — cost × solve rate by group
# ══════════════════════════════════════════════════════════════════════════════
# Three groups:
#   cheap-only  : S(x) ∩ {reflect,replan} ≠ ∅  AND  escalate ∉ S(x)
#   both-work   : S(x) ∩ {reflect,replan} ≠ ∅  AND  escalate ∈ S(x)
#   esc-only    : S(x) = {escalate}

def group(ex):
    sa = set(ex.get("successful_actions", []))
    cheap = sa & {"reflect", "replan"}
    if "escalate" in sa and cheap:
        return "both"
    if "escalate" in sa and not cheap:
        return "esc_only"
    if cheap and "escalate" not in sa:
        return "cheap_only"
    return "other"

groups = {"cheap_only": [], "both": [], "esc_only": []}
for ex in examples:
    g = group(ex)
    if g in groups:
        groups[g].append(ex)

def stats_for(exlist, strategy):
    """strategy: 'escalate' | 'oracle' (cheapest in S(x))"""
    costs, solved = [], []
    for ex in exlist:
        sa = set(ex.get("successful_actions", []))
        if strategy == "escalate":
            action = "escalate"
        else:  # oracle: cheapest in S(x)
            action = min(sa, key=lambda a: get_cost(ex, a))
        costs.append(get_cost(ex, action))
        solved.append(1 if action in sa else 0)
    return np.mean(costs) * 1000, np.mean(solved)   # cost in m$

group_labels = ["Cheap-recovery\nonly solvable\n(18.6%)",
                "Both cheap &\nescalate work\n(36.1%)",
                "Escalate-only\nsolvable\n(45.0%)"]
group_keys   = ["cheap_only", "both", "esc_only"]

esc_costs,    esc_solves    = [], []
oracle_costs, oracle_solves = [], []
for k in group_keys:
    ec, es = stats_for(groups[k], "escalate")
    oc, os = stats_for(groups[k], "oracle")
    esc_costs.append(ec);    esc_solves.append(es)
    oracle_costs.append(oc); oracle_solves.append(os)

x = np.arange(3)
w = 0.32

fig, (ax_cost, ax_solve) = plt.subplots(1, 2, figsize=(8, 3.4))

# -- cost panel --
bars_esc    = ax_cost.bar(x - w/2, esc_costs,    w, color=C_ESC,   label="Always-escalate", zorder=3)
bars_oracle = ax_cost.bar(x + w/2, oracle_costs, w, color=C_CHEAP, label="Oracle routing",   zorder=3)
ax_cost.set_xticks(x); ax_cost.set_xticklabels(group_labels, fontsize=8.5)
ax_cost.set_ylabel("Mean cost (m\$)", fontsize=9)
ax_cost.yaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
ax_cost.set_axisbelow(True)
ax_cost.spines[["top","right"]].set_visible(False)
ax_cost.set_title("Mean cost per example", fontsize=10)
ax_cost.legend(fontsize=8, framealpha=0.9, edgecolor=C_GRID)
for bar in bars_esc:
    ax_cost.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars_oracle:
    ax_cost.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)

# -- solve rate panel --
bars_esc2    = ax_solve.bar(x - w/2, esc_solves,    w, color=C_ESC,   label="Always-escalate", zorder=3)
bars_oracle2 = ax_solve.bar(x + w/2, oracle_solves, w, color=C_CHEAP, label="Oracle routing",   zorder=3)
ax_solve.set_xticks(x); ax_solve.set_xticklabels(group_labels, fontsize=8.5)
ax_solve.set_ylabel("Solve rate", fontsize=9)
ax_solve.set_ylim(0, 1.12)
ax_solve.yaxis.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
ax_solve.set_axisbelow(True)
ax_solve.spines[["top","right"]].set_visible(False)
ax_solve.set_title("Solve rate", fontsize=10)
for bar in bars_esc2:
    ax_solve.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars_oracle2:
    ax_solve.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Always-escalate vs. oracle routing, by solvability group  ($n=720$)",
             fontsize=10, y=1.01)
plt.tight_layout()
out_b = OUT / "fig_oracle_vs_escalate.pdf"
plt.savefig(out_b, bbox_inches="tight")
plt.savefig(str(out_b).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
plt.close()
print(f"saved → {out_b}")
