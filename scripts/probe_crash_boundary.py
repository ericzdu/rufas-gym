#!/usr/bin/env python3
"""Stage 1 probe — where does a concentrated ration actually crash RuFaS?

`implementers.LOGIT_SCALE` was cut 5.0 -> 1.5 on the strength of a comment: "feeds #50 and
#301 above ~60-80% drive the cow's nitrogen balance negative". That cut is what put the
CMA-ES optimum outside PPO's reach, so before replacing it with per-feed caps the boundary
is worth *measuring* rather than inheriting.

For each candidate feed and each target inclusion share, this builds a lactating-cow ration
holding that share and spreading the remainder evenly, then runs a short episode and
records whether RuFaS survived. The output is the highest share that is reliably safe,
which becomes `MAX_INCLUSION` in `implementers.py`.

Episodes run through `RufasEnv`, so each gets a fresh subprocess and the in-process
slowdown does not accumulate across the sweep.

    python scripts/probe_crash_boundary.py                     # feeds 50 and 301
    python scripts/probe_crash_boundary.py --feeds 50 --years 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rufas_rl import EnvConfig, RufasEnv
from rufas_rl.implementers import RationImplementer

# High enough that an extreme target share is actually expressible; the probe sets
# percentages directly and encodes them, so it must not be clipped by a low scale.
PROBE_LOGIT_SCALE = 8.0
FAILURE_PENALTY = -1e7


def ration_with_share(sizes, groups, feed_ids_per_group, target_feed, share):
    """Percentages per group: `target_feed` at `share`, the rest of that group even.

    Groups that do not contain the target feed are left at an even split, so the probe
    isolates one feed in one ration rather than perturbing the whole farm.
    """
    out = []
    for size, ids in zip(sizes, feed_ids_per_group):
        if target_feed in ids and size > 1:
            pct = [(100.0 - share * 100.0) / (size - 1)] * size
            pct[ids.index(target_feed)] = share * 100.0
        else:
            pct = [100.0 / size] * size
        out.append(pct)
    return out


def run(env, action, steps):
    """One episode; returns the step it failed on, or None if it survived."""
    env.reset(seed=42)
    for step in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        if info.get("simulation_failed"):
            return step
        if terminated or truncated:
            break
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", type=int, nargs="*", default=[50, 301])
    ap.add_argument("--shares", type=float, nargs="*",
                    default=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    ap.add_argument("--years", type=int, default=1)
    ap.add_argument("--out", default="results/crash_boundary.json")
    args = ap.parse_args()

    steps = args.years * 12
    config = EnvConfig(
        cadence="monthly", max_steps=steps, rewarder="profit",
        levers=("rations",), failure_penalty=FAILURE_PENALTY,
    )
    env = RufasEnv(config)
    spec = env.scenario_spec
    implementer = RationImplementer(spec, logit_scale=PROBE_LOGIT_SCALE)

    # Which feed IDs sit in which ration, straight off the scenario's feed file.
    from rufas_rl.bootstrap import resolve
    feed_blob = json.loads(resolve(
        json.loads(spec.scenario_metadata_path.read_text())["files"]["feed"]["path"]
    ).read_text())
    feed_ids_per_group = [[f["feed_type"] for f in r["feeds"]] for r in feed_blob["rations"]]

    print(f"Crash-boundary probe — {args.years}-yr episodes ({steps} steps), "
          f"logit_scale={PROBE_LOGIT_SCALE}")
    print(f"  ration groups: {list(spec.ration_groups)}")
    for g, ids in zip(spec.ration_groups, feed_ids_per_group):
        print(f"    {g:<10} {ids}")
    print()

    results = []
    t0 = time.time()
    for feed in args.feeds:
        present = [g for g, ids in zip(spec.ration_groups, feed_ids_per_group) if feed in ids]
        if not present:
            print(f"feed {feed}: not in any ration, skipping")
            continue
        print(f"feed {feed} (in {present}):")
        for share in args.shares:
            pct = ration_with_share(spec.ration_sizes, spec.ration_groups,
                                    feed_ids_per_group, feed, share)
            action = implementer.encode(pct)
            realized = implementer.decode(action)
            # encode() clips into the action box, so report what was actually achieved.
            achieved = max(
                r[ids.index(feed)]
                for r, ids in zip(realized, feed_ids_per_group) if feed in ids
            )
            failed_at = run(env, action, steps)
            status = "CRASH at step %d" % failed_at if failed_at is not None else "survived"
            print(f"  target {share:>5.0%}  achieved {achieved:>5.1f}%  -> {status}")
            results.append({"feed": feed, "target_share": share,
                            "achieved_pct": achieved, "failed_at": failed_at})
        print()

    env.close()

    print("=" * 60)
    for feed in args.feeds:
        rows = [r for r in results if r["feed"] == feed]
        safe = [r for r in rows if r["failed_at"] is None]
        crashed = [r for r in rows if r["failed_at"] is not None]
        if not rows:
            continue
        hi = max((r["achieved_pct"] for r in safe), default=None)
        lo = min((r["achieved_pct"] for r in crashed), default=None)
        print(f"feed {feed}: highest safe {hi if hi is None else f'{hi:.1f}%'}, "
              f"lowest crash {lo if lo is None else f'{lo:.1f}%'}")
    print(f"({(time.time() - t0) / 60:.1f} min)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "years": args.years, "logit_scale": PROBE_LOGIT_SCALE, "results": results,
    }, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
