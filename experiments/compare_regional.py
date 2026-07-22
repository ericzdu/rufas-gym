#!/usr/bin/env python3
"""Evaluate RuFaS's built-in regional default rations as fixed-ration baselines.

RuFaS ships four regional feed/ration defaults — Midwest, Northeast, Southwest,
Intermountain — each a diet built for that region's feed availability and prices. This
runs the same farm (same herd, fields, weather) under each regional ration and records
profit, so the learned RL policy can be placed against all of them, not just the Midwest
default it was trained on.

Each regional ration is injected at construction (its whole feed blob replaces the
scenario's), then the full horizon runs and profit = milk revenue - RuFaS's own feed cost
is read from the output pool. One fresh subprocess per region (no in-process degradation).

    python experiments/compare_regional.py --years 7 --out results/regional.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from rufas_rl.bootstrap import resolve, rufas_cwd  # noqa: E402
from rufas_rl.rewarders import DEFAULT_MILK_PRICE  # noqa: E402
from rufas_rl.runtime import flush_singletons, prepare_metadata, start_kwargs  # noqa: E402

REGIONS = {
    "Midwest": "input/data/feed/example_Midwest_feed.json",
    "Northeast": "input/data/feed/example_Northeast_feed.json",
    "Southwest": "input/data/feed/example_Southwest_feed.json",
    "Intermountain": "input/data/feed/example_Intermountain_feed.json",
}


def _sum(v):
    if isinstance(v, bool):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return sum(_sum(x) for x in v)
    if isinstance(v, dict):
        return sum(_sum(x) for x in v.values())
    return 0.0


def evaluate_region(region: str, years: int) -> dict:
    """Run the horizon under a region's default ration; return profit components."""
    from RUFAS.output_manager import OutputManager
    from RUFAS.task_manager import TaskManager

    regional_feed = json.loads(resolve(REGIONS[region]).read_text())
    end = f"{2013 + years - 1}:365"

    flush_singletons()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        meta = prepare_metadata("input/task_manager_metadata.json", work, seed=42)
        orig = TaskManager._handle_simulation_engine_run_tasks

        def handler(args, im, om, tid, pg, flush):
            # Replace the feed blob outright. input_patch would DEEP-MERGE it, combining
            # this region's rations element-wise with the scenario's Midwest ones (their
            # percentages then sum past 100 and the ration is rejected). So set the pool
            # entry directly and only patch the end date through input_patch.
            im._InputManager__pool["feed"] = regional_feed
            args["input_patch"] = {"config": {"end_date": end}}
            return orig(args, im, om, tid, pg, flush)

        TaskManager._handle_simulation_engine_run_tasks = staticmethod(handler)
        try:
            TaskManager().start(metadata_path=meta, **start_kwargs(work))
            pool = OutputManager()._get_flat_variables_pool()
            errs = getattr(OutputManager(), "errors_pool", {}) or {}
            failed = any("Failed to finish" in k or "ammonia" in k.lower() for k in errs)
        finally:
            TaskManager._handle_simulation_engine_run_tasks = orig

    milk = sum(_sum(v) for k, v in pool.items() if "milk_production" in k.lower())
    feed_cost = sum(_sum(v) for k, v in pool.items()
                    if "ration_interval" in k and k.endswith("_cost"))
    revenue = milk * DEFAULT_MILK_PRICE
    return {"region": region, "failed": failed, "milk_kg": milk,
            "feed_cost": feed_cost, "revenue": revenue, "profit": revenue - feed_cost}


def main() -> None:
    # Internal per-region subprocess entry — checked before argparse so its positional
    # args don't confuse the parser.
    if len(sys.argv) > 1 and sys.argv[1] == "--run-region":
        region, years, out_path = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
        with rufas_cwd():
            out_path.write_text(json.dumps(evaluate_region(region, years)))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=7)
    ap.add_argument("--out", default="results/regional.json")
    args = ap.parse_args()

    results = {}
    tmp = Path(tempfile.mkdtemp())
    print(f"{'region':>14} {'profit':>14} {'feed cost':>14} {'milk (t)':>10}")
    for region in REGIONS:
        p = tmp / f"{region}.json"
        subprocess.run([sys.executable, str(THIS), "--run-region", region, str(args.years), str(p)],
                       check=True)
        r = json.loads(p.read_text())
        results[region] = r
        flag = "  FAILED" if r["failed"] else ""
        print(f"{region:>14} {r['profit']:>14,.0f} {r['feed_cost']:>14,.0f} "
              f"{r['milk_kg']/1000:>10,.0f}{flag}")

    Path(args.out).write_text(json.dumps({"years": args.years, "regions": results}, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
