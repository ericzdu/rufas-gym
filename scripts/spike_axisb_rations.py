#!/usr/bin/env python3
"""
Axis-B equivalence spike — RATIONS (monthly's defining mid-year lever).

Central question for online stepping: since RuFaS caches the ration config at
construction (RationManager class attrs; the pool is never re-read), can we change
rations MID-RUN by live-mutating that class state, and is that EQUIVALENT to having
the ration in the pool at construction (the only proven injection path)?

Three modes, each run in a FRESH SUBPROCESS (so the in-process 5x leak can't
confound the comparison), each dumping variables_pool to JSON:

  baseline  — default ration, no change.
  construct — modified ration injected at CONSTRUCTION via input_patch deep-merge
              (the proven path).
  mutate    — default ration in the pool, but RationManager live-mutated to the
              modified ration at the first formulate_rations call (day 0).

The SAME modification (reverse each group's ration_percentage vector — sum-preserving
and feasible) is applied in construct and mutate.

PASS (Axis-B proven for rations):
  * construct  DIFFERS from baseline   (the modification actually matters), AND
  * mutate  ==  construct              (live mid-run mutation == construction inject).

Usage:
  driver:  ../RuFaS/venv/bin/python scripts/spike_axisb_rations.py
  (the driver spawns the three subprocess runs, then compares)
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()


def rufas_root() -> Path:
    env = os.environ.get("RUFAS_ROOT")
    return Path(env).resolve() if env else (THIS.parent.parent.parent / "RuFaS").resolve()


def ensure_importable(root: Path) -> None:
    if (root / "RUFAS").is_dir():
        sys.path.insert(0, str(root))
    import RUFAS  # noqa: F401


def patched_metadata(root: Path, work_dir: Path,
                     metadata_rel: str = "input/task_manager_metadata.json") -> Path:
    meta = json.loads((root / metadata_rel).read_text())
    tasks_rel = Path(meta["files"]["tasks"]["path"])
    tasks_file = tasks_rel if tasks_rel.is_absolute() else root / tasks_rel
    tasks = json.loads(tasks_file.read_text())
    tasks["parallel_workers"] = 1
    for t in tasks.get("tasks", []):
        if t.get("task_type") == "SIMULATION_MULTI_RUN":
            t["multi_run_counts"] = 1
    patched_tasks = work_dir / "patched_tasks.json"
    patched_tasks.write_text(json.dumps(tasks, indent=2))
    meta["files"]["tasks"] = {**meta["files"]["tasks"], "path": str(patched_tasks.resolve())}
    patched_meta = work_dir / "patched_metadata.json"
    patched_meta.write_text(json.dumps(meta, indent=2))
    return patched_meta


def reverse_rations(feed_blob: dict) -> dict:
    """Return a deep copy of the feed blob with each group's ration_percentage
    vector reversed across its feeds (sum-preserving, feed_types unchanged)."""
    mod = copy.deepcopy(feed_blob)
    for ration in mod["rations"]:
        pcts = [f["ration_percentage"] for f in ration["feeds"]]
        for f, p in zip(ration["feeds"], reversed(pcts)):
            f["ration_percentage"] = p
    return mod


# ---------------------------------------------------------------------------
# One simulation run (in this process), with the mode's monkey-patch installed.
# ---------------------------------------------------------------------------
def run_one(mode: str, out_path: Path) -> None:
    root = rufas_root()
    ensure_importable(root)
    from RUFAS.input_manager import InputManager
    from RUFAS.output_manager import LogVerbosity, OutputManager
    from RUFAS.task_manager import TaskManager
    from RUFAS.util import Utility
    from RUFAS.biophysical.animal.herd_manager import HerdManager

    prev = Path.cwd()
    os.chdir(root)
    try:
        InputManager().flush_pool()
        OutputManager().flush_pools()

        with tempfile.TemporaryDirectory(prefix="axisb_") as tmp:
            work_dir = Path(tmp)
            meta = patched_metadata(root, work_dir)

            orig_handler = TaskManager._handle_simulation_engine_run_tasks

            def handler(args, input_manager, output_manager, task_id,
                        produce_graphics, should_flush_im_pool):
                if mode == "construct":
                    feed = input_manager.get_data("feed")
                    mod = reverse_rations(feed)
                    args["input_patch"] = {"feed": {"rations": mod["rations"]}}
                return orig_handler(args, input_manager, output_manager, task_id,
                                    produce_graphics, should_flush_im_pool)

            TaskManager._handle_simulation_engine_run_tasks = staticmethod(handler)

            if mode == "mutate":
                from RUFAS.biophysical.animal.ration.ration_manager import RationManager
                orig_formulate = HerdManager.formulate_rations
                state = {"done": False}

                def formulate(self, *a, **kw):
                    if not state["done"]:
                        feed = InputManager().get_data("feed")
                        mod = reverse_rations(feed)
                        RationManager.set_user_defined_rations(mod)
                        RationManager.set_user_defined_ration_tolerance(mod)
                        state["done"] = True
                    return orig_formulate(self, *a, **kw)

                HerdManager.formulate_rations = formulate

            TaskManager().start(
                metadata_path=meta,
                verbosity=LogVerbosity("none"),
                exclude_info_maps=True,
                output_directory=work_dir / "output",
                logs_directory=work_dir / "output" / "logs",
                clear_output_directory=False,
                produce_graphics=False,
                suppress_log_files=True,
                metadata_depth_limit=None,
            )

            pool = OutputManager()._get_flat_variables_pool()
            out_path.write_text(json.dumps(_jsonable(pool)))
            print(f"[{mode}] variables_pool keys: {len(pool)} -> {out_path}")
    finally:
        os.chdir(prev)


def _jsonable(obj):
    """Best-effort convert the flat pool to JSON (values may be lists of numbers)."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return {k: _coerce(v) for k, v in obj.items()} if isinstance(obj, dict) else str(obj)


def _coerce(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


# ---------------------------------------------------------------------------
# Comparison of the three dumped pools.
# ---------------------------------------------------------------------------
# Keys that log the INPUT we injected (what the setter was called with), not a
# simulation RESULT. They differ trivially because `mutate` calls the setter twice
# (default at construction, then our reversed ration) vs once in `construct`.
INSTRUMENTATION = ("set_user_defined_rations", "set_ration_feeds",
                   "set_user_defined_ration_tolerance")


def _is_instrumentation(key: str) -> bool:
    return any(tag in key for tag in INSTRUMENTATION)


def _flatten_numbers(pool: dict) -> dict[str, float]:
    """Map each PHYSICAL key to the sum of its numeric leaves (a scalar fingerprint)."""
    out = {}
    for k, v in pool.items():
        if _is_instrumentation(k):
            continue
        nums = _numbers(v)
        if nums:
            out[k] = sum(nums)
    return out


def _numbers(v):
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        acc = []
        for x in v:
            acc += _numbers(x)
        return acc
    if isinstance(v, dict):
        acc = []
        for x in v.values():
            acc += _numbers(x)
        return acc
    return []


def _rel_diff(a: dict[str, float], b: dict[str, float]) -> tuple[int, list[tuple[str, float, float]]]:
    keys = set(a) | set(b)
    n_diff = 0
    worst: list[tuple[str, float, float]] = []
    for k in keys:
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        denom = max(abs(av), abs(bv), 1e-9)
        rd = abs(av - bv) / denom
        if rd > 1e-6:
            n_diff += 1
            worst.append((k, rd, av - bv))
    worst.sort(key=lambda t: -t[1])
    return n_diff, worst[:8]


def compare(paths: dict[str, Path]) -> None:
    pools = {m: _flatten_numbers(json.loads(p.read_text())) for m, p in paths.items()}
    print("\n================ Axis-B rations spike ================")
    for m, fp in pools.items():
        print(f"  {m:9s}: {len(fp)} numeric keys")

    cb_n, cb_w = _rel_diff(pools["construct"], pools["baseline"])
    mc_n, mc_w = _rel_diff(pools["mutate"], pools["construct"])
    mb_n, _ = _rel_diff(pools["mutate"], pools["baseline"])

    print(f"\n  construct vs baseline : {cb_n} keys differ  (expect >0 — modification matters)")
    for k, rd, d in cb_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")
    print(f"  mutate    vs construct: {mc_n} keys differ  (expect 0 — live-mutate == construct)")
    for k, rd, d in mc_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")
    print(f"  mutate    vs baseline : {mb_n} keys differ  (expect >0)")

    passed = cb_n > 0 and mc_n == 0
    print("\n  RESULT: " + (
        "PASS — live mid-run ration mutation == construction injection. "
        "Axis-B proven for rations; O(T) online stepping of rations is viable."
        if passed else
        "FAIL / INCONCLUSIVE — see diffs above. If construct==baseline, the "
        "modification didn't take (strengthen it). If mutate!=construct, live "
        "mutation is NOT equivalent (timing/state issue) — monthly ration control "
        "needs a different mechanism."))
    print("======================================================\n")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--run":  # internal: single-mode subprocess entry
        run_one(args[1], Path(args[2]))
        return

    # Driver: spawn the three runs as fresh subprocesses, then compare.
    py = sys.executable
    out_dir = Path(tempfile.mkdtemp(prefix="axisb_out_"))
    paths = {m: out_dir / f"{m}.json" for m in ("baseline", "construct", "mutate")}
    for m, p in paths.items():
        print(f"--- running {m} (fresh subprocess) ---")
        subprocess.run([py, str(THIS), "--run", m, str(p)], check=True)
    compare(paths)


if __name__ == "__main__":
    main()
