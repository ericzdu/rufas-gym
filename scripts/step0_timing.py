#!/usr/bin/env python3
"""
Step 0 — feasibility gate (V1).

Times ONE full multi-year RuFaS run on the base scenario, in-process with
workers==1 (the path the RL env will actually use), and prints per-simulated-year
and total wall-clock. This single number selects the stepping strategy:

    sub-second/yr  -> replay is fine, do the simple thing
    seconds/yr     -> stepping patch + subprocess parallelism
    minutes/yr     -> online RL is dead; pivot to offline / surrogate

No RL code, no RuFaS source edits. Mirrors RuFaS `main.py`'s TaskManager.start
call, but forces parallel_workers=1 (in-process) as runner.py does.

Run with the RuFaS venv (has the deps):
    ../RuFaS/venv/bin/python scripts/step0_timing.py
or point at a checkout elsewhere with RUFAS_ROOT=/path/to/RuFaS.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path


def rufas_root() -> Path:
    env = os.environ.get("RUFAS_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parent.parent.parent / "RuFaS").resolve()


def ensure_importable(root: Path) -> None:
    """Make the SOURCE checkout's RUFAS win over any (stale/partial) installed copy."""
    if (root / "RUFAS").is_dir():
        sys.path.insert(0, str(root))  # front of path -> shadows site-packages
    try:
        import RUFAS  # noqa: F401
    except ModuleNotFoundError as err:
        raise SystemExit(
            f"Cannot import RuFaS from {root} (exists={root.exists()}).\n"
            "Use the RuFaS venv:  ../RuFaS/venv/bin/python scripts/step0_timing.py\n"
            "Or set RUFAS_ROOT to the checkout."
        ) from err
    loaded = Path(RUFAS.__file__).resolve().parent
    if root not in loaded.parents:
        print(f"WARNING: imported RUFAS from {loaded}, not the checkout under {root}",
              file=sys.stderr)


def derive_sim_years(root: Path, config_rel: str = "input/data/config/example_freestall_config.json") -> int | None:
    """simulation_length_years = end_year - start_year + 1 (rufas_time.py:28)."""
    cfg = root / config_rel
    if not cfg.exists():
        return None
    data = json.loads(cfg.read_text())
    try:
        start = int(str(data["start_date"]).split(":")[0])
        end = int(str(data["end_date"]).split(":")[0])
        return end - start + 1
    except (KeyError, ValueError):
        return None


def patched_metadata(root: Path, work_dir: Path,
                     metadata_rel: str = "input/task_manager_metadata.json") -> Path:
    """Copy the default metadata/tasks, forcing parallel_workers=1 (in-process)."""
    meta_file = root / metadata_rel
    meta = json.loads(meta_file.read_text())

    tasks_rel = Path(meta["files"]["tasks"]["path"])
    tasks_file = tasks_rel if tasks_rel.is_absolute() else root / tasks_rel
    tasks = json.loads(tasks_file.read_text())
    tasks["parallel_workers"] = 1
    for task in tasks.get("tasks", []):
        if task.get("task_type") == "SIMULATION_MULTI_RUN":
            task["multi_run_counts"] = 1

    patched_tasks = work_dir / "patched_tasks.json"
    patched_tasks.write_text(json.dumps(tasks, indent=2))

    meta["files"]["tasks"] = {**meta["files"]["tasks"], "path": str(patched_tasks.resolve())}
    patched_meta = work_dir / "patched_metadata.json"
    patched_meta.write_text(json.dumps(meta, indent=2))
    return patched_meta


def time_one_run(root: Path, metadata_path: Path, work_dir: Path) -> float:
    """Run one full simulation in-process, chdir'd into the RuFaS root, timed."""
    from RUFAS.input_manager import InputManager
    from RUFAS.output_manager import LogVerbosity, OutputManager
    from RUFAS.task_manager import TaskManager

    # RuFaS reads/writes relative paths (input/, output/); run from its root.
    prev = Path.cwd()
    os.chdir(root)
    try:
        # Flush BOTH singletons before each call. NOTE: this is NOT sufficient —
        # run 2 is still ~5x slower than run 1 even with both pools flushed AND a
        # fresh output dir. There is an in-memory global-state leak somewhere in
        # RuFaS; in-process reuse degrades. The harness must run each RuFaS
        # invocation in a FRESH SUBPROCESS (consistent with RuFaS's own
        # maxtasksperchild=1 for workers>1). See README "Step 0".
        try:
            InputManager().flush_pool()
            OutputManager().flush_pools()
        except Exception as e:
            print(f"  (flush warning: {e})", file=sys.stderr)
        t0 = time.perf_counter()
        TaskManager().start(
            metadata_path=metadata_path,
            verbosity=LogVerbosity("none"),
            exclude_info_maps=True,
            output_directory=work_dir / "output",
            logs_directory=work_dir / "output" / "logs",
            clear_output_directory=False,
            produce_graphics=False,
            suppress_log_files=True,
            metadata_depth_limit=None,
        )
        dt = time.perf_counter() - t0
        # Sanity: confirm the run did real work (non-empty variables_pool), so the
        # timing reflects a real simulation, not a fast-fail path.
        try:
            pool = OutputManager()._get_flat_variables_pool()
            n_keys = len(pool) if isinstance(pool, dict) else -1
        except Exception:
            n_keys = -1
        print(f"    (variables_pool keys: {n_keys})", file=sys.stderr)
        return dt
    finally:
        os.chdir(prev)


def main() -> None:
    root = rufas_root()
    ensure_importable(root)
    years = derive_sim_years(root)

    with tempfile.TemporaryDirectory(prefix="step0_") as tmp:
        work_dir = Path(tmp)
        meta = patched_metadata(root, work_dir)

        print(f"RuFaS root: {root}")
        print(f"simulation_length_years (from config): {years if years else 'unknown'}")
        print("Timing one full in-process run (workers=1)...\n")

        # Two runs: first pays import/JIT/IO warmup, second is the steadier number.
        # Fresh output dir per run isolates disk-accumulation from in-memory leaks.
        durations = []
        for i in range(2):
            run_out = work_dir / f"run{i}"
            dt = time_one_run(root, meta, run_out)
            durations.append(dt)
            per_year = f"{dt / years:.3f} s/sim-year" if years else "n/a"
            print(f"  run {i + 1}: {dt:8.3f} s total   ({per_year})")

        best = min(durations)
        print("\n--- Step 0 result ---")
        print(f"total wall-clock (best of {len(durations)}): {best:.3f} s")
        if years:
            pyr = best / years
            print(f"per-simulated-year:                    {pyr:.3f} s")
            band = ("sub-second -> replay OK" if pyr < 1
                    else "seconds -> stepping patch + parallelism" if pyr < 60
                    else "minutes -> offline / surrogate")
            print(f"strategy band:                         {band}")
        print("\nRecord this number in the README (V1).")


if __name__ == "__main__":
    main()
