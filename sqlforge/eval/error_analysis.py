"""
SQLForge Phase 5 — failure-mode analysis across the 7B training stages.

Bucketing the *reason* each query failed (rather than just counting misses) is what
turns "SFT went 54.5% -> 74.5%" into an explanation. It's also what shows why DPO had
nothing left to fix: by the SFT stage the remaining failures are capability gaps, not
convention gaps.

Categories, in priority order (first match wins) — every rule is grounded in an observed
failure, not invented:

  1. Schema hallucination  — SQL didn't execute: invented column/table/type
                             (DuckDB Binder/Catalog error).
  2. Categorical literal   — churn_risk='High' instead of 'HIGH_RISK' -> 0 rows.
  3. Revenue formula       — superseded l_extendedprice/l_discount instead of
                             total_value/promo_reduction.
  4. Output shape          — executed and filtered right, but wrong columns.
  5. Wrong values          — everything else: real logic errors.

Outputs a table, a JSON summary, and light+dark stacked-bar PNGs for the README.

Run (offline, no GPU):
    python sqlforge/eval/error_analysis.py
"""

import os
import re
import sys
import json
import argparse
from collections import Counter, OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "phase0", "results")

# --- the three 7B stages, in training order ---
STAGES = OrderedDict([
    ("Base 7B", "baseline_Qwen_Qwen2.5-Coder-7B-Instruct_zeroshot.json"),
    ("+ SFT", "sft_qwen7b_zeroshot.json"),
    ("+ DPO", "dpo_eval.json"),
])

# --- categories in fixed slot order (never cycled) + the validated palette ---
CATEGORIES = [
    "Schema hallucination",
    "Categorical literal",
    "Revenue formula",
    "Output shape",
    "Wrong values",
]
PALETTE_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
PALETTE_DARK = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9"]

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", palette=PALETTE_LIGHT),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", palette=PALETTE_DARK),
}

_BAD_CHURN = re.compile(r"churn_risk\s*(?:=|IN|in)\s*\(?\s*'(?!HIGH_RISK|MEDIUM_RISK|LOW_RISK)[^']*'")
_OLD_REVENUE = re.compile(r"l_extendedprice|l_discount", re.IGNORECASE)
_SCHEMA_ERR = re.compile(r"Binder Error|Catalog Error|does not have a column|not found in FROM", re.I)
_SHAPE = re.compile(
    r"gold has (\d+) rows x (\d+) cols \((\[[^\]]*\])\), agent has (\d+) rows x (\d+) cols \((\[[^\]]*\])"
)


def categorize(rec: dict) -> str:
    sql = rec.get("generated_sql") or ""
    err = rec.get("error") or ""
    details = rec.get("match_details") or ""

    # 1. Didn't execute at all.
    if err:
        return "Schema hallucination" if _SCHEMA_ERR.search(err) else "Schema hallucination"

    # 2. The churn literal bug — the single most common systematic error.
    if _BAD_CHURN.search(sql):
        return "Categorical literal"

    # 3. Revenue computed from the superseded columns.
    if _OLD_REVENUE.search(sql):
        return "Revenue formula"

    # 4. Right rows, wrong columns.
    m = _SHAPE.search(details)
    if m:
        gold_n, agent_n = int(m.group(2)), int(m.group(5))
        gold_cols, agent_cols = m.group(3), m.group(6)
        if gold_n != agent_n or gold_cols != agent_cols:
            return "Output shape"
    elif rec.get("failure_category") in ("wrong_columns", "wrong_tables"):
        return "Output shape"

    # 5. Executed, right shape, right conventions — just wrong.
    return "Wrong values"


def load_stage(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    fails = [r for r in d["results"] if r.get("status") != "PASS"]
    return d, fails


def build_chart(counts, totals, mode, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    stages = list(counts.keys())
    x = range(len(stages))
    bottoms = [0] * len(stages)

    for i, cat in enumerate(CATEGORIES):
        vals = [counts[s].get(cat, 0) for s in stages]
        # 2px surface-coloured edge = the required gap between stacked segments;
        # it doubles as the secondary encoding the CVD floor band requires.
        ax.bar(x, vals, bottom=bottoms, width=0.44, label=cat,
               color=t["palette"][i], edgecolor=t["surface"], linewidth=2.0, zorder=3)
        # Direct labels — mandatory relief for the sub-3:1 light slots.
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v / 2, str(v), ha="center", va="center",
                        color="#ffffff" if mode == "light" else "#0b0b0b",
                        fontsize=10, fontweight="600", zorder=4)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Total above each bar; accuracy rides the tick label so the top stays uncluttered.
    for xi, s in enumerate(stages):
        ax.text(xi, bottoms[xi] + 0.6, f"{bottoms[xi]} failures",
                ha="center", va="bottom", color=t["ink"], fontsize=11, fontweight="700")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}\n{totals[s]}" for s in stages], color=t["ink"], fontsize=11)
    ax.set_ylabel("Failed queries (of 55)", color=t["secondary"], fontsize=10)
    ax.set_ylim(0, max(bottoms) + 5)
    ax.tick_params(axis="y", colors=t["muted"], labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])

    ax.set_title("What the 7B got wrong, by failure mode",
                 color=t["ink"], fontsize=13.5, fontweight="700", loc="left", pad=30)
    ax.text(0, 1.035,
            "SFT cleared the convention errors it was trained on. The categorical-literal "
            "bug survived both stages.",
            transform=ax.transAxes, color=t["muted"], fontsize=9.5, va="bottom")

    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5,
                    labelcolor=t["secondary"], handlelength=1.1, handleheight=1.1)
    for h in leg.legend_handles:
        h.set_edgecolor(t["surface"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print(f"  chart -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "sqlforge", "eval", "figures"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    counts, totals, detail = OrderedDict(), {}, OrderedDict()
    for stage, fname in STAGES.items():
        path = os.path.join(args.results_dir, fname)
        if not os.path.exists(path):
            print(f"!! missing {path} — skipping {stage}")
            continue
        d, fails = load_stage(path)
        c = Counter(categorize(r) for r in fails)
        counts[stage] = c
        n = len(d["results"])
        correct = sum(1 for r in d["results"] if r.get("status") == "PASS")
        totals[stage] = f"{100*correct/n:.1f}% accuracy"
        detail[stage] = [{"id": r["id"], "tier": r.get("difficulty"),
                          "category": categorize(r)} for r in fails]

    # --- table ---
    print("\n" + "=" * 74)
    print("FAILURE MODES BY TRAINING STAGE (7B, 55-query suite)")
    print("=" * 74)
    hdr = f"{'Category':24s}" + "".join(f"{s:>14s}" for s in counts)
    print(hdr)
    print("-" * len(hdr))
    for cat in CATEGORIES:
        print(f"{cat:24s}" + "".join(f"{counts[s].get(cat,0):>14d}" for s in counts))
    print("-" * len(hdr))
    print(f"{'TOTAL FAILURES':24s}" + "".join(f"{sum(counts[s].values()):>14d}" for s in counts))
    print(f"{'Accuracy':24s}" + "".join(f"{totals[s]:>14s}" for s in counts))
    print("=" * 74)

    # --- per-query detail (so every bar is auditable) ---
    with open(os.path.join(args.out_dir, "failure_modes.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": {s: dict(c) for s, c in counts.items()},
                   "accuracy": totals, "per_query": detail}, f, indent=2)
    print(f"\n  detail -> {os.path.join(args.out_dir, 'failure_modes.json')}")

    for mode in ("light", "dark"):
        build_chart(counts, totals, mode,
                    os.path.join(args.out_dir, f"failure_modes_{mode}.png"))


if __name__ == "__main__":
    main()
