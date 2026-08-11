#!/usr/bin/env python3
"""Evaluate the non-learned baselines under exactly the environment PPO trains on.

Every arm here runs through `train_ppo.make_env_config`, on the same horizon, the same
seeds and therefore the same price paths. That is deliberate: the single largest problem
with the first result was that PPO and CMA-ES were quietly measured on different problems
(different levers, different action space, different eval seed, different discounting), so
the 12% gap was not attributable to anything in particular.

The arms:

* **configured** — the farm's actual ration, held fixed. The "can we beat current
  practice?" reference.
* **neutral** — an even split, held fixed. A control, not a serious competitor.
* **greedy(sharpness)** — `baselines.GreedyPriceRation`, swept over sharpness. This is the
  real bar. Profit here is close to a linear program in the ration with time-varying
  coefficients, so a policy that simply tilts toward this month's cheapest feeds captures
  much of the gain that is available at all. PPO has to beat the *best* greedy setting,
  not the first one tried.

Fixed-ration arms cannot respond to prices by construction, so the gap between the best of
them and the best greedy setting measures how much price-responsiveness is worth here
before any learning is involved.

    python experiments/eval_baselines.py --years 2 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ppo import configured_action, make_env_config  # noqa: E402

from rufas_rl import RufasEnv, load_spec  # noqa: E402
from rufas_rl.baselines import GreedyPriceRation  # noqa: E402

SHARPNESSES = (0.0, 0.25, 0.5, 0.75, 1.0)


def episode_price_path(env, seed: int):
    """Rebuild, in this process, the exact price path the episode will run under.

    Needed to keep the comparison fair. `info["prices"]` reports the interval just
    *scored*, so a policy driven off it would always be a month stale — it would be
    choosing this month's ration from last month's prices, while PPO's observation carries
    the current month. That lag would understate the greedy baseline and flatter PPO.

    Price paths are pure functions of the episode seed (`Episode.start` builds them that
    way precisely so this is possible), so reconstructing one here reproduces the worker's
    path exactly rather than approximating it.
    """
    from rufas_rl.prices import make_price_path

    spec = env.scenario_spec
    return make_price_path(
        spec,
        process=env.config.price_process,
        n_months=spec.n_years * 12,
        seed=seed,
        levels=env.config.price_levels,
        **env.config.price_kwargs,
    )


def run_episode(env, policy, steps: int, seed: int) -> dict:
    """One episode. `policy(prices) -> action`, given the upcoming month's prices.

    At monthly cadence on a January-start scenario, decision `k` governs month `k`, which
    is the index `Episode` itself derives from the simulation calendar — so `path.at(k)`
    is the same vector the episode applies on that step.
    """
    env.reset(seed=seed)
    path = episode_price_path(env, seed)
    totals = {"profit": 0.0, "feed_cost": 0.0, "milk_revenue": 0.0, "milk_kg": 0.0}

    for step in range(steps):
        action = policy(path.at(step))
        _, _, terminated, truncated, info = env.step(action)
        if info.get("simulation_failed"):
            return {"failed": True, "failed_at": step, **totals}
        for key in totals:
            totals[key] += info[key]
        # Cross-check that the reconstructed path matches what the episode actually
        # applied; a mismatch would silently invalidate every baseline number.
        priced = info.get("prices")
        if priced and not np.isclose(priced["milk"], path.at(step).milk, rtol=1e-9):
            raise AssertionError(
                f"Reconstructed price path diverged at step {step}: episode applied "
                f"milk={priced['milk']}, reconstruction has {path.at(step).milk}"
            )
        if terminated or truncated:
            break
    return {"failed": False, **totals}


def evaluate(env, policy, steps: int, seeds) -> dict:
    """Mean profit over `seeds`; each seed is a different price path and herd draw."""
    runs = [run_episode(env, policy, steps, seed) for seed in seeds]
    ok = [r for r in runs if not r["failed"]]
    if not ok:
        return {"mean_profit": float("-inf"), "n_failed": len(runs), "runs": runs}
    return {
        "mean_profit": float(np.mean([r["profit"] for r in ok])),
        "std_profit": float(np.std([r["profit"] for r in ok])),
        "mean_feed_cost": float(np.mean([r["feed_cost"] for r in ok])),
        "mean_milk_kg": float(np.mean([r["milk_kg"] for r in ok])),
        "n_failed": len(runs) - len(ok),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=3, help="seeds per arm")
    ap.add_argument("--price-process", default="synthetic",
                    choices=["static", "synthetic", "real"])
    ap.add_argument("--price-levels", default="realistic",
                    choices=["configured", "realistic"])
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    steps = args.years * 12
    # Held-out seeds, distinct from the 0..n_envs range training uses.
    seeds = [1000 + i for i in range(args.episodes)]

    config = make_env_config(args.years, price_process=args.price_process,
                             price_levels=args.price_levels)
    env = RufasEnv(config)
    spec = load_spec()

    print(f"Baselines — {args.years}-yr, {args.episodes} seeds, "
          f"prices={args.price_process}/{args.price_levels}")
    print(f"  seeds: {seeds}\n")

    arms: dict[str, object] = {}
    t0 = time.time()

    fixed = {
        "configured": configured_action(spec),
        "neutral": env.neutral_action(),
    }
    for name, action in fixed.items():
        result = evaluate(env, lambda _prices, a=action: a, steps, seeds)
        arms[name] = result
        print(f"  {name:<18} ${result['mean_profit']:>13,.0f} "
              f"(+/-{result.get('std_profit', 0):>11,.0f})  failed={result['n_failed']}")

    print()
    for sharpness in SHARPNESSES:
        greedy = GreedyPriceRation(spec, sharpness=sharpness)
        result = evaluate(env, greedy.action, steps, seeds)
        arms[f"greedy_{sharpness}"] = result
        print(f"  greedy s={sharpness:<10.2f} ${result['mean_profit']:>13,.0f} "
              f"(+/-{result.get('std_profit', 0):>11,.0f})  failed={result['n_failed']}")

    env.close()

    greedy_arms = {k: v for k, v in arms.items() if k.startswith("greedy")}
    best_greedy = max(greedy_arms, key=lambda k: greedy_arms[k]["mean_profit"])
    best_fixed = max(("configured", "neutral"), key=lambda k: arms[k]["mean_profit"])

    print("\n" + "=" * 70)
    print(f"best fixed ration : {best_fixed} "
          f"${arms[best_fixed]['mean_profit']:,.0f}")
    print(f"best greedy       : {best_greedy} "
          f"${greedy_arms[best_greedy]['mean_profit']:,.0f}")
    value = greedy_arms[best_greedy]["mean_profit"] - arms[best_fixed]["mean_profit"]
    print(f"value of price-following, before any learning: ${value:+,.0f}")
    print("This is the bar PPO has to clear to have shown anything.")
    print(f"({(time.time() - t0) / 60:.1f} min)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "years": args.years, "episodes": args.episodes, "seeds": seeds,
        "price_process": args.price_process, "price_levels": args.price_levels,
        "arms": arms, "best_fixed": best_fixed, "best_greedy": best_greedy,
        "value_of_price_following": value,
    }, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
