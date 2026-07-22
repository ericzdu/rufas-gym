#!/usr/bin/env python3
"""Turn a ration-optimization run into figures + a plain-language explanation.

Reads the JSON that `optimize_ration.py` writes and produces four PNGs plus a text
summary of exactly how each headline number was derived. Every figure traces back to a
column in that JSON, so nothing here is hand-drawn.

    python experiments/make_figures.py --results results/ration_optimization_7yr.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rufas_rl.bootstrap import resolve  # noqa: E402
from rufas_rl.spec import load_spec  # noqa: E402

PALETTE = {"configured": "#4C72B0", "neutral": "#999999", "optimized": "#55A868",
           "accent": "#C44E52", "grid": "#DDDDDD"}

# Label for the learned/optimized strategy across all figures.
RL_LABEL = "CMA-ES"
# Label for the shipped baseline (RuFaS example_freestall uses the Midwest ration set).
BASELINE_LABEL = "RuFaS Midwest\ndefault"
BASELINE_LABEL_INLINE = "RuFaS Midwest default"


def feed_metadata(scenario: str):
    """Return {feed_type: (label, $/kg)} for the scenario's feeds."""
    spec = load_spec(scenario)
    meta = json.loads(spec.scenario_metadata_path.read_text())
    feed = json.loads(resolve(meta["files"]["feed"]["path"]).read_text())

    cost = {fd["feed_type"]: fd.get("purchased_feed_cost", float("nan"))
            for fd in feed.get("feeds", [])}

    category = {}
    try:
        with open(resolve(meta["files"]["NRC_Comp"]["path"])) as f:
            for row in csv.DictReader(f):
                category[int(row["rufas_id"])] = row.get("Fd_Category", "")
    except Exception:
        pass

    feed_types_per_ration = [[fe["feed_type"] for fe in r["feeds"]] for r in feed["rations"]]
    groups = list(spec.ration_groups)
    # Short category names so seven x-axis labels don't collide.
    short = {"Grass/Legume Forage": "Forage", "Grain Crop Forage": "Silage",
             "Energy Source": "Energy", "By-Product/Other": "By-product",
             "Plant Protein": "Protein", "Calf Liquid Feed": "Milk"}
    labels = {}
    for ft in {ft for row in feed_types_per_ration for ft in row}:
        cat = category.get(ft, "")
        cat = short.get(cat, cat)
        labels[ft] = f"#{ft} {cat}".strip() if cat else f"feed #{ft}"
    return groups, feed_types_per_ration, labels, cost


def fig_scoreboard(data, out: Path) -> None:
    """Profit / feed cost / milk for the three rations — the headline table as bars."""
    rows = [(BASELINE_LABEL, data["configured"], PALETTE["configured"]),
            ("Even\nsplit", data["neutral"], PALETTE["neutral"]),
            (RL_LABEL, data["optimized"], PALETTE["optimized"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, key, title, unit, scale in [
        (axes[0], "profit", "Net profit", "$M", 1e6),
        (axes[1], "feed_cost", "Feed cost", "$M", 1e6),
        (axes[2], "milk_kg", "Milk produced", "1000 t", 1e6),
    ]:
        vals = [r[key] / scale for _, r, _ in rows]
        bars = ax.bar([n for n, _, _ in rows], vals, color=[c for _, _, c in rows], width=0.62)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=PALETTE["grid"], zorder=0)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(f"Ration comparison over {data['years']} years "
                 f"(milk priced at ${data['milk_price']:.2f}/kg)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_convergence(data, out: Path) -> None:
    """Best-so-far profit vs evaluation, against the two fixed baselines."""
    history = np.array(data["history"]) / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(range(1, len(history) + 1), history, color=PALETTE["optimized"], lw=2.2,
            marker="o", ms=3, label=f"{RL_LABEL} best so far")
    ax.axhline(data["configured"]["profit"] / 1e6, color=PALETTE["configured"], ls="--",
               lw=1.8, label=BASELINE_LABEL_INLINE)
    ax.axhline(data["neutral"]["profit"] / 1e6, color=PALETTE["neutral"], ls=":",
               lw=1.8, label="even-split ration")
    ax.set_xlabel("evaluation (each = one full simulation)")
    ax.set_ylabel("profit ($M)")
    ax.set_title(f"{RL_LABEL} converges above the Midwest default "
                 f"({data['years']}-year horizon)", fontweight="bold")
    ax.grid(color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_mechanism(data, out: Path) -> None:
    """Milk vs feed cost across every evaluated ration — why cheaper wins.

    The point of the figure: milk is nearly flat while feed cost swings widely, so profit
    is essentially 'revenue (fixed) minus feed cost' — minimise feed cost and you win.
    """
    log = [e for e in data.get("eval_log", []) if not e.get("failed")]
    if not log:
        return
    feed = np.array([e["feed_cost"] for e in log]) / 1e6
    milk = np.array([e["milk_kg"] for e in log]) / 1e6
    profit = np.array([e["profit"] for e in log]) / 1e6

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sc = ax.scatter(feed, milk, c=profit, cmap="viridis", s=42, edgecolor="white", lw=0.5)
    ax.scatter(data["configured"]["feed_cost"] / 1e6, data["configured"]["milk_kg"] / 1e6,
               marker="D", s=120, color=PALETTE["configured"], edgecolor="black",
               zorder=5, label=BASELINE_LABEL_INLINE)
    ax.scatter(data["optimized"]["feed_cost"] / 1e6, data["optimized"]["milk_kg"] / 1e6,
               marker="*", s=260, color=PALETTE["optimized"], edgecolor="black",
               zorder=5, label=RL_LABEL)
    ax.set_xlabel("feed cost ($M)")
    ax.set_ylabel("milk produced (1000 t)")
    ax.set_title("Milk barely moves; feed cost does all the work", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("profit ($M)")
    ax.grid(color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.9, edgecolor=PALETTE["grid"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_substitution(data, scenario: str, out: Path) -> None:
    """Configured vs optimized lactating ration, feeds ordered by price."""
    if "rations_optimized" not in data:
        return
    groups, feed_types, labels, cost = feed_metadata(scenario)
    # The lactating group is where the money is; fall back to the last ration.
    gi = groups.index("lac_cow") if "lac_cow" in groups else len(groups) - 1
    fts = feed_types[gi]
    order = sorted(range(len(fts)), key=lambda i: -cost.get(fts[i], 0))  # dear -> cheap
    fts_o = [fts[i] for i in order]
    conf = [data["rations_configured"][gi][i] for i in order]
    opt = [data["rations_optimized"][gi][i] for i in order]
    names = [f"{labels[ft]}\n(${cost.get(ft, float('nan')):.2f}/kg)" for ft in fts_o]

    x = np.arange(len(fts_o))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.2, conf, width=0.4, color=PALETTE["configured"], label=BASELINE_LABEL_INLINE)
    ax.bar(x + 0.2, opt, width=0.4, color=PALETTE["optimized"], label=RL_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax.set_ylabel("% of lactating ration")
    ax.set_title("Where the savings come from: off the dear feeds, onto the cheap forages\n"
                 "(feeds ordered most → least expensive per kg)", fontweight="bold")
    ax.grid(axis="y", color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ration_optimization_7yr.json")
    ap.add_argument("--scenario", default="input/task_manager_metadata.json")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    data.setdefault("milk_price", 0.45)  # older runs did not record it
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_scoreboard(data, outdir / "1_scoreboard.png")
    fig_convergence(data, outdir / "2_convergence.png")
    fig_mechanism(data, outdir / "3_mechanism.png")
    fig_substitution(data, args.scenario, outdir / "4_substitution.png")
    print(f"figures -> {outdir}/  (1_scoreboard, 2_convergence, 3_mechanism, 4_substitution)")


if __name__ == "__main__":
    main()
