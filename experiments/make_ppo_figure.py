#!/usr/bin/env python3
"""Plot the PPO learning curve and its comparison to the baselines.

    python experiments/make_ppo_figure.py --result results/ppo_smoke/result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {"rl": "#3F7A5B", "neutral": "#999999", "grid": "#DDDDDD", "crash": "#A8503F"}


def moving_average(y, w=10):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="results/ppo_smoke/result.json")
    ap.add_argument("--out", default="results/figures/5_ppo_learning.png")
    args = ap.parse_args()

    data = json.loads(Path(args.result).read_text())
    curve = data.get("curve", [])
    if not curve:
        print("no learning curve in result; nothing to plot")
        return

    t = np.array([c["t"] for c in curve])
    r = np.array([c["return"] for c in curve])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.scatter(t, r, s=14, color=PALETTE["rl"], alpha=0.35, label="episode return")
    if len(r) >= 10:
        ma = moving_average(r, 10)
        ax.plot(t[len(t) - len(ma):], ma, color=PALETTE["rl"], lw=2.4,
                label="10-episode moving average")

    neutral_ret = data["neutral"]["mean_return"]
    ax.axhline(neutral_ret, color=PALETTE["neutral"], ls="--", lw=1.8,
               label=f"neutral policy ({neutral_ret:.1f})")

    ax.set_xlabel("training timesteps")
    ax.set_ylabel("episode return (profit, scaled)")
    ax.set_title(f"PPO learning curve — {data['years']}-year episodes, "
                 f"{data['timesteps']:,} steps, {data['train_min']:.0f} min",
                 fontweight="bold")
    ax.grid(color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure -> {out}")

    print(f"\nlearned mean profit ${data['learned']['mean_profit']:,.0f}  "
          f"vs neutral ${data['neutral']['mean_profit']:,.0f}  "
          f"(gain ${data['gain_vs_neutral']:+,.0f})")


if __name__ == "__main__":
    main()
