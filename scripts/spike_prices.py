#!/usr/bin/env python3
"""Stage 0 spike — does repricing feed actually move RuFaS?

The unit tests in `rufas_rl/tests/test_prices.py` only check that the price *paths* are
sane. They cannot check the claim the whole reformulation rests on:

    writing `Feed.purchase_cost` mid-run changes what the simulator does.

That claim has two halves, and this spike separates them, because only one of them makes
prices worth adding:

  * **Accounting.** Reported feed cost must move with prices. Necessary but boring — a
    post-hoc multiplication would achieve the same.
  * **Dynamics.** RuFaS formulates rations by least cost (`RationOptimizer.objective` is
    `sum(amounts * price_list)`, re-solved every 30 days within 10% of the ration we set).
    So a price change should also alter the *physical* quantities purchased. If only the
    accounting moves, prices are a cosmetic multiplier and the experiment is dead.

Four checks:

  1. REGRESSION  — `price_process="static"` reproduces a pre-price-lever episode exactly.
  2. PROPAGATION — one write reaches every live reference to the feed (in-process).
  3. ACCOUNTING  — doubling a feed's price raises reported cost for that feed.
  4. DYNAMICS    — a large relative price shift changes kg purchased, not just dollars.

RESULT, 2-year horizon, neutral ration (2026-07-27): 1, 2 and 3 PASS; **4 FAILS**.
Swinging relative prices 30x moved the purchased kilograms of all ten feeds by exactly
0.00%; the entire +$13.8k cost change is the same physical purchases repriced (predicted
+$14.2k from kg x price deltas). RuFaS's least-cost formulation is pinned at the
all-lower-bounds corner of its +/-10% box, which is price-independent because every cost
coefficient is positive. So prices are accounting-only with respect to RuFaS, and the
agent's ration is the sole price-responsive channel. That is a clean result rather than a
broken one — see `rufas_rl/prices.py` for what it means for the experiment — but it is
the reason a greedy price-following heuristic has to be one of the baselines.

Checks 1, 3 and 4 each run real multi-year simulations, so this takes a while (~15 min at
2 years; episodes run in-process here and hit the known ~5x reuse slowdown).

    python scripts/spike_prices.py            # all four, 2-year horizon
    python scripts/spike_prices.py --years 1  # faster
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from rufas_rl import EnvConfig
from rufas_rl.bootstrap import rufas_cwd
from rufas_rl.episode import Episode
from rufas_rl.prices import PriceVector, StaticPricePath
from rufas_rl.spec import load_spec

FAILURE_DOLLARS = -1e7
PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"


def _config(years: int, **kwargs) -> EnvConfig:
    return EnvConfig(
        cadence="monthly",
        max_steps=years * 12,
        rewarder="profit",
        levers=("rations",),
        failure_penalty=FAILURE_DOLLARS,
        **kwargs,
    )


def _purchases_by_feed(episode: Episode) -> dict[int, float]:
    """kg dry matter purchased per feed so far, from RuFaS's own output pool.

    `FeedManager.purchase_feed` reports `ration_interval_<id>_amount_purchased` alongside
    the cost, which is what makes the dynamics check possible at all — we can see
    quantities, not only dollars.
    """
    from RUFAS.output_manager import OutputManager

    pool = OutputManager()._get_flat_variables_pool()
    out: dict[int, float] = defaultdict(float)
    for key, value in pool.items():
        if "ration_interval" not in key or not key.endswith("_amount_purchased"):
            continue
        for part in key.split("_"):
            if part.isdigit():
                out[int(part)] += _sum_numbers(value)
                break
    return dict(out)


def _sum_numbers(value) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, dict):
        return sum(_sum_numbers(v) for v in value.values())
    return 0.0


def run_episode(config: EnvConfig, action, steps: int) -> dict:
    """One episode at a fixed action. Returns totals plus per-feed kg purchased."""
    episode = Episode(config)
    try:
        episode.start(seed=42)
        totals = defaultdict(float)
        for _ in range(steps):
            _, _, terminated, truncated, info = episode.step(action)
            if info.get("simulation_failed"):
                return {"failed": True}
            for key in ("profit", "feed_cost", "milk_revenue", "milk_kg"):
                totals[key] += info[key]
            if terminated or truncated:
                break
        totals["purchased_kg"] = _purchases_by_feed(episode)
        totals["failed"] = False
        return dict(totals)
    finally:
        episode.close()


# -- 1. regression --------------------------------------------------------------------


def check_regression(years: int, action) -> bool:
    """Static prices must reproduce the environment as it behaved before this change.

    Run twice — once with the price machinery engaged on the scenario's own constants,
    once with the applier disabled entirely — and require bit-identical profit.
    """
    print("\n[1] REGRESSION — static prices reproduce the un-priced environment")
    steps = years * 12

    with_prices = run_episode(_config(years, price_process="static"), action, steps)

    # The control: same episode with the price applier neutered, i.e. RuFaS running on
    # exactly the prices its JSON configured, untouched.
    config = _config(years, price_process="static")
    episode = Episode(config)
    episode.price_applier = _NullApplier()
    try:
        episode.start(seed=42)
        control = defaultdict(float)
        for _ in range(steps):
            _, _, terminated, truncated, info = episode.step(action)
            if info.get("simulation_failed"):
                print(f"  {FAIL} control episode failed")
                return False
            for key in ("profit", "feed_cost", "milk_kg"):
                control[key] += info[key]
            if terminated or truncated:
                break
    finally:
        episode.close()

    ok = True
    for key in ("profit", "feed_cost", "milk_kg"):
        a, b = with_prices[key], control[key]
        match = np.isclose(a, b, rtol=0, atol=1e-6)
        ok &= match
        print(f"  {'ok ' if match else 'DIFF'} {key:<13} applied={a:>16,.4f}  control={b:>16,.4f}")
    print(f"  {PASS if ok else FAIL} static prices are a no-op")
    return bool(ok)


class _NullApplier:
    lever = "prices"

    def apply(self, prices) -> None:
        pass


# -- 2. propagation -------------------------------------------------------------------


def check_propagation(years: int) -> bool:
    """One write must reach every live reference to a feed.

    RuFaS builds a single `list[Feed]` at construction and hands the same list to the feed
    manager, the herd manager and every pen. This asserts that sharing is real — if some
    consumer ever takes a copy, prices would silently stop reaching the ration optimizer
    and the experiment would quietly revert to being static.
    """
    print("\n[2] PROPAGATION — one write reaches every holder of the feed list")
    from rufas_rl.appliers import PriceApplier, set_engine

    episode = Episode(_config(years, price_process="static"))
    try:
        episode.start(seed=42)
        engine = episode._engine
        set_engine(engine)

        feed_manager = engine.feed_manager
        engine_feeds = {f.rufas_id: f for f in engine.available_feeds}
        manager_feeds = {f.rufas_id: f for f in feed_manager._available_feeds}

        shared = all(engine_feeds[i] is manager_feeds[i] for i in engine_feeds)
        print(f"  {'ok ' if shared else 'DIFF'} feed manager shares the engine's Feed objects: {shared}")

        target = 44  # corn grain
        before = engine_feeds[target].purchase_cost
        PriceApplier().apply(PriceVector(milk=0.45, feeds={target: before * 3.0}))
        after_engine = engine_feeds[target].purchase_cost
        after_manager = manager_feeds[target].purchase_cost

        moved = np.isclose(after_engine, before * 3.0) and np.isclose(after_manager, before * 3.0)
        print(f"  {'ok ' if moved else 'DIFF'} feed {target}: {before} -> {after_engine} "
              f"(manager sees {after_manager})")

        # The path that actually reaches the least-cost optimizer. No object stores the
        # feed list: `simulation_engine` passes `self.available_feeds` down to
        # `HerdManager.formulate_rations`, which narrows it per pen via
        # `_find_pen_available_feeds` — a list comprehension over the *same* Feed objects.
        # `RationConfig` then reads `feed.purchase_cost` off them at solve time. So the
        # thing to assert is that the narrowing preserves identity; if it ever started
        # copying, prices would silently stop reaching the optimizer.
        ration_ids = [f.rufas_id for f in engine.available_feeds]
        narrowed = engine.herd_manager._find_pen_available_feeds(
            engine.available_feeds, ration_ids
        )
        identity_ok = all(n is engine_feeds[n.rufas_id] for n in narrowed)
        optimizer_price = next(
            (f.purchase_cost for f in narrowed if f.rufas_id == target), None
        )
        pens_ok = identity_ok and np.isclose(optimizer_price, before * 3.0)
        print(f"  {'ok ' if identity_ok else 'DIFF'} pen narrowing preserves object identity "
              f"({len(narrowed)} feeds)")
        print(f"  {'ok ' if pens_ok else 'DIFF'} optimizer would price feed {target} at "
              f"{optimizer_price}")

        # An unnamed feed must be left alone — that is how partial price vectors work.
        other = engine_feeds[301].purchase_cost
        untouched = np.isclose(other, 1.0)
        print(f"  {'ok ' if untouched else 'DIFF'} unnamed feed 301 untouched at {other}")

        ok = bool(shared and moved and pens_ok and untouched)
        print(f"  {PASS if ok else FAIL} price writes propagate")
        return ok
    finally:
        episode.close()


# -- 3 & 4. accounting and dynamics ---------------------------------------------------


def check_accounting_and_dynamics(years: int, action) -> tuple[bool, bool]:
    """Reprice the forage/concentrate ratio and watch both dollars and kilograms.

    The shock is deliberately a *relative* one — forage up 10x, corn grain down 3x —
    because RuFaS's least-cost formulation only responds to which feed is cheapest, not
    to the overall price level. It also happens to be the direction that corrects the
    scenario's broken placeholder prices, so it is the shift the real experiment makes.
    """
    print("\n[3/4] ACCOUNTING + DYNAMICS — a relative price shift moves dollars and kg")
    steps = years * 12
    spec = load_spec("input/task_manager_metadata.json", 0)
    base = dict(spec.feed_prices)

    shocked = dict(base)
    for forage in (50, 95, 104, 110):
        shocked[forage] = base[forage] * 10.0
    shocked[44] = base[44] / 3.0

    baseline = run_episode(_config(years, price_process="static"), action, steps)
    shock = _run_with_prices(_config(years), action, steps, shocked)

    if baseline.get("failed") or shock.get("failed"):
        print(f"  {FAIL} an episode failed; cannot compare")
        return False, False

    cost_moved = not np.isclose(baseline["feed_cost"], shock["feed_cost"], rtol=1e-9)
    print(f"  {'ok ' if cost_moved else 'DIFF'} feed cost  "
          f"{baseline['feed_cost']:>14,.0f} -> {shock['feed_cost']:>14,.0f}  "
          f"({100 * (shock['feed_cost'] / baseline['feed_cost'] - 1):+.1f}%)")
    print(f"      profit {baseline['profit']:>14,.0f} -> {shock['profit']:>14,.0f}")

    print("\n  kg dry matter purchased, per feed:")
    b_kg, s_kg = baseline["purchased_kg"], shock["purchased_kg"]
    qty_moved = False
    for fid in sorted(set(b_kg) | set(s_kg)):
        b, s = b_kg.get(fid, 0.0), s_kg.get(fid, 0.0)
        delta = 100 * (s / b - 1) if b else float("nan")
        flag = ""
        if b and not np.isclose(b, s, rtol=1e-6):
            qty_moved = True
            flag = "  <-- moved"
        print(f"    feed {fid:>3}  {b:>14,.0f} -> {s:>14,.0f}  ({delta:+7.2f}%){flag}")

    print(f"\n  {PASS if cost_moved else FAIL} ACCOUNTING — reported cost tracks prices")
    if qty_moved:
        print(f"  {PASS} DYNAMICS — physical purchases changed; RuFaS re-optimized")
    else:
        print(f"  {FAIL} DYNAMICS — quantities identical. Prices are only cosmetic here:")
        print("       the least-cost tweak is bounded by user_defined_ration_tolerance")
        print("       (10%) and may be pinned by binding nutrition constraints. The")
        print("       agent's own ration lever still works, but RuFaS will not help.")
    return cost_moved, qty_moved


def _run_with_prices(config: EnvConfig, action, steps: int, feed_prices: dict) -> dict:
    """One episode held at a fixed, non-scenario price vector."""
    from rufas_rl.rewarders import DEFAULT_MILK_PRICE

    episode = Episode(config)
    try:
        episode.start(seed=42)
        # Override *after* start(), which builds the path from config and would clobber
        # an earlier assignment.
        episode.price_path = StaticPricePath(milk=DEFAULT_MILK_PRICE, feeds=feed_prices)
        totals = defaultdict(float)
        for _ in range(steps):
            _, _, terminated, truncated, info = episode.step(action)
            if info.get("simulation_failed"):
                return {"failed": True}
            for key in ("profit", "feed_cost", "milk_revenue", "milk_kg"):
                totals[key] += info[key]
            if terminated or truncated:
                break
        totals["purchased_kg"] = _purchases_by_feed(episode)
        totals["failed"] = False
        return dict(totals)
    finally:
        episode.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()

    spec = load_spec("input/task_manager_metadata.json", 0)
    from rufas_rl.implementers import RationImplementer

    action = RationImplementer(spec).neutral_action()

    print("=" * 74)
    print(f"Stage 0 price spike — {args.years}-year episodes, neutral ration")
    print("=" * 74)

    results = {}
    # RuFaS resolves its input paths relative to its own repo root, and `Episode` is being
    # driven in-process here rather than through the env's worker subprocess (which does
    # this for us).
    with rufas_cwd():
        if not args.skip_regression:
            results["regression"] = check_regression(args.years, action)
        results["propagation"] = check_propagation(args.years)
        accounting, dynamics = check_accounting_and_dynamics(args.years, action)
    results["accounting"] = accounting
    results["dynamics"] = dynamics

    print("\n" + "=" * 74)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
    print("=" * 74)

    # Dynamics failing is informative, not fatal — it would mean the agent's lever is the
    # only price-responsive channel. The others failing means the wiring is broken.
    critical = [v for k, v in results.items() if k != "dynamics"]
    return 0 if all(critical) else 1


if __name__ == "__main__":
    sys.exit(main())
