#!/usr/bin/env python3
"""Experiment 1 — optimizing the ration for profit in RuFaS-gym.

The question this answers, in one sentence an advisor can hold onto:

    Given RuFaS's own feed costs and milk, can an optimizer find a ration that makes more
    money than the farm's configured ration — and if so, how?

Setup:
  * Environment: `RufasEnv` with the `profit` reward — milk revenue (kg x an exogenous
    milk price) minus RuFaS's *own* reported feed cost. The cost side is entirely RuFaS's
    economics; only the milk price is external, and the ranking of rations is insensitive
    to it (a cheaper ration at equal milk wins at any positive price).
  * Objective: total profit over a `--years`-year horizon (default 2, for fast evals).
  * Optimizer: CMA-ES over the full ration action, **seeded at the farm's configured
    ration** so "can we beat the farm?" is asked directly.
  * Baselines it is compared against: the configured ration (the farm's actual choice)
    and an even-split ration.

Every evaluation is one fresh-subprocess episode (the env's rule). Rations that crash
RuFaS's manure chemistry — extreme diets do — are scored at a large loss via
`failure_penalty`, so the optimizer learns to avoid that region rather than dying on it.

    python experiments/optimize_ration.py --years 2 --budget 60 --out results/ration.json

Numbers are cold-cache wall-clock on the dev Mac; treat magnitudes as order-of-magnitude.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rufas_rl import EnvConfig, RufasEnv
from rufas_rl.bootstrap import rufas_cwd
from rufas_rl.implementers import RationImplementer
from rufas_rl.rewarders import DEFAULT_MILK_PRICE

# A ration that fails is worth a big loss so the optimizer avoids that corner of the box.
FAILURE_DOLLARS = -1e7


def configured_action(spec) -> np.ndarray:
    """The scenario's actual ration, expressed as an action vector.

    Read straight from the scenario's feed JSON — the same file `spec` was built from —
    so this needs no RuFaS import and no running simulation.
    """
    scenario_meta = json.loads(spec.scenario_metadata_path.read_text())
    from rufas_rl.bootstrap import resolve

    feed = json.loads(resolve(scenario_meta["files"]["feed"]["path"]).read_text())
    pcts = [[f["ration_percentage"] for f in r["feeds"]] for r in feed["rations"]]
    return RationImplementer(spec).encode(pcts)


def evaluate(env: RufasEnv, action: np.ndarray, steps: int) -> dict:
    """Run one episode holding `action` fixed; return total profit and its components."""
    env.reset(seed=42)
    profit = feed_cost = revenue = milk_kg = 0.0
    failed = False
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        if info.get("simulation_failed"):
            failed = True
            break
        profit += info["profit"]
        feed_cost += info["feed_cost"]
        revenue += info["milk_revenue"]
        milk_kg += info["milk_kg"]
        if terminated or truncated:
            break
    if failed:
        return {"profit": FAILURE_DOLLARS, "feed_cost": float("nan"),
                "revenue": float("nan"), "milk_kg": float("nan"), "failed": True}
    return {"profit": profit, "feed_cost": feed_cost, "revenue": revenue,
            "milk_kg": milk_kg, "failed": False}


def optimize(
    env: RufasEnv, x0: np.ndarray, steps: int, budget: int, seed: int, popsize: int = 8
) -> tuple[np.ndarray, list, list]:
    import cma

    history: list[float] = []   # best-so-far, per eval (convergence curve)
    eval_log: list[dict] = []   # every evaluation's full telemetry (mechanism plots)
    best_x, best_profit = x0.copy(), -np.inf
    n_evals = 0

    es = cma.CMAEvolutionStrategy(
        list(x0), 0.4,
        {"bounds": [-1.0, 1.0], "seed": seed, "popsize": popsize, "verbose": -9},
    )
    # Evaluate whole generations — CMA needs the full population in `tell`, so budget is
    # effectively rounded up to a multiple of popsize.
    while not es.stop() and n_evals < budget:
        solutions = es.ask()
        losses = []
        for x in solutions:
            result = evaluate(env, np.asarray(x, dtype=np.float32), steps)
            n_evals += 1
            losses.append(-result["profit"])  # CMA minimizes
            if result["profit"] > best_profit:
                best_profit, best_x = result["profit"], np.asarray(x, dtype=np.float32)
            history.append(best_profit)
            eval_log.append({"eval": n_evals, **result})
            print(f"  eval {n_evals:>3}/{budget}  profit=${result['profit']:>13,.0f}"
                  f"  best=${best_profit:>13,.0f}" + ("  (FAILED)" if result["failed"] else ""))
        es.tell(solutions, losses)
    return best_x, history, eval_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2, help="horizon in simulated years")
    ap.add_argument("--budget", type=int, default=60, help="CMA-ES evaluation budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="input/task_manager_metadata.json")
    ap.add_argument("--out", default="results/ration_optimization.json")
    args = ap.parse_args()

    steps = args.years * 12
    config = EnvConfig(
        task_metadata_path=args.scenario,
        cadence="monthly",
        max_steps=steps,
        rewarder="profit",
        failure_penalty=FAILURE_DOLLARS,
    )
    env = RufasEnv(config)
    spec = env.scenario_spec

    print(f"Scenario: {spec.simulation_type}, {args.years}-year horizon ({steps} steps), "
          f"action dim {env.action_space.shape[0]}, budget {args.budget} evals\n")

    x_configured = configured_action(spec)
    x_neutral = env.neutral_action()

    t0 = time.time()
    print("Baseline — farm's configured ration:")
    r_configured = evaluate(env, x_configured, steps)
    print(f"  profit=${r_configured['profit']:,.0f}  feed=${r_configured['feed_cost']:,.0f}"
          f"  milk={r_configured['milk_kg']/1000:,.0f} t\n")

    print("Baseline — even-split ration:")
    r_neutral = evaluate(env, x_neutral, steps)
    print(f"  profit=${r_neutral['profit']:,.0f}  feed=${r_neutral['feed_cost']:,.0f}"
          f"  milk={r_neutral['milk_kg']/1000:,.0f} t\n")

    print(f"CMA-ES optimizing profit (seeded at configured ration):")
    x_best, history, eval_log = optimize(env, x_configured, steps, args.budget, args.seed)
    r_best = evaluate(env, x_best, steps)
    env.close()
    elapsed = time.time() - t0

    # Decode the two rations to percentages so the figures can show *what changed*.
    implementer = RationImplementer(spec)
    rations_configured = implementer.decode(x_configured)
    rations_optimized = implementer.decode(x_best)

    gain = r_best["profit"] - r_configured["profit"]
    pct = 100.0 * gain / abs(r_configured["profit"]) if r_configured["profit"] else float("nan")
    feed_cut = r_configured["feed_cost"] - r_best["feed_cost"]
    milk_change = 100.0 * (r_best["milk_kg"] - r_configured["milk_kg"]) / max(r_configured["milk_kg"], 1)

    print("\n" + "=" * 68)
    print(f"RESULT ({args.years}-year horizon, {args.budget} evals, {elapsed/60:.1f} min)")
    print("=" * 68)
    print(f"{'ration':>22} {'profit ($)':>15} {'feed cost ($)':>15} {'milk (t)':>10}")
    for name, r in [("configured (farm)", r_configured), ("even split", r_neutral),
                    ("CMA-ES optimized", r_best)]:
        print(f"{name:>22} {r['profit']:>15,.0f} {r['feed_cost']:>15,.0f} "
              f"{r['milk_kg']/1000:>10,.0f}")
    print("-" * 68)
    print(f"Optimized vs configured: profit {gain:+,.0f} ({pct:+.1f}%), "
          f"feed cost {-feed_cut:+,.0f}, milk {milk_change:+.1f}%")
    print("Interpretation: the gain is a feed-substitution effect — RuFaS milk is largely")
    print("insensitive to ration composition, so profit-optimal feeding shifts toward the")
    print("cheapest feeds that maintain intake.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "years": args.years, "budget": args.budget, "seed": args.seed,
        "milk_price": DEFAULT_MILK_PRICE, "steps": steps,
        "elapsed_min": elapsed / 60,
        "configured": r_configured, "neutral": r_neutral, "optimized": r_best,
        "gain_dollars": gain, "gain_pct": pct,
        "best_action": x_best.tolist(), "history": history, "eval_log": eval_log,
        "ration_groups": list(spec.ration_groups),
        "rations_configured": rations_configured,
        "rations_optimized": rations_optimized,
    }, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
