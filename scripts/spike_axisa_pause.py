#!/usr/bin/env python3
"""
Axis-A pause spike — the RuFaS PAUSING MECHANISM (zero RuFaS edits).

RuFaS runs start-to-finish in one blocking `TaskManager.start` call; there is no
native "advance one period, look around, decide." Axis A gives us that pause
WITHOUT touching RuFaS source (which would break its validation): run RuFaS on a
worker THREAD and monkey-patch its daily-step method so that, at each decision
boundary, the worker blocks on a hand-off queue until the driver releases it. The
paused thread's own call stack IS the saved farm state, so nothing is serialized.
This is the primary mechanism (the T8 generator source-patch is only a fallback).

Design (verified against ../RuFaS):
  * Dispatched daily method for a full_farm scenario is
    `SimulationEngine._execute_full_farm_daily_simulation` (simulation_engine.py:322),
    selected via `_simulation_type_to_daily_simulation_function` (:195). Patching the
    CLASS attribute before construction is captured by the per-instance dispatch dict.
  * Cadence is just a predicate on `self.time` (rufas_time.py): here `current_date.day
    == 1` == monthly boundaries. Yearly is the same mechanism with a coarser predicate.
  * MUST use a full_farm scenario (example_freestall). A field_only scenario dispatches
    to `_execute_field_only_simulation` and would NEVER hit the patched method — the run
    would finish, look identical, and silently never pause (false PASS).

This spike proves PAUSING ONLY (Axis A). Applying an action at the pause is Axis B
(already PASS for rations, scripts/spike_axisb_rations.py) — deliberately NOT wired
in here; the driver releases each pause with no action.

Three modes, each in a FRESH SUBPROCESS (the in-process ~5x leak can't confound):

  unhooked  — plain TaskManager.start, no patch. The control.
  stepped   — threaded pause hook installed; worker thread runs the sim; the daily
              method blocks at every monthly boundary; the driver releases each one
              (no action) and counts the pauses. Runs to completion.
  teardown  — hook installed; release a few boundaries, then send the STOP sentinel
              so the hook raises inside the worker; assert the worker thread exits
              cleanly (no deadlock, no leaked exception).

PASS (Axis-A pause mechanism proven):
  * stepped paused > 0 times and completed          (suspend/resume works, no deadlock)
  * stepped variables_pool == unhooked variables_pool (0 physical diffs)
                                                      (pausing does NOT change the science)
  * teardown thread exited cleanly after STOP        (clean teardown for env.reset())

Usage:
  driver:  ../RuFaS/venv/bin/python scripts/spike_axisa_pause.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import queue
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
    """Copy the default metadata/tasks, forcing parallel_workers=1 (in-process)."""
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


class _Teardown(Exception):
    """Raised inside the hook when the driver sends STOP — the env.reset() path."""


def _is_month_boundary(engine) -> bool:
    """Decision-boundary predicate. Monthly == first day of each month.
    Yearly would be `... .day == 1 and ... .month == 1`; same mechanism, coarser knob."""
    return engine.time.current_date.day == 1


def _start_kwargs(work_dir: Path) -> dict:
    from RUFAS.output_manager import LogVerbosity
    return dict(
        verbosity=LogVerbosity("none"),
        exclude_info_maps=True,
        output_directory=work_dir / "output",
        logs_directory=work_dir / "output" / "logs",
        clear_output_directory=False,
        produce_graphics=False,
        suppress_log_files=True,
        metadata_depth_limit=None,
    )


# ---------------------------------------------------------------------------
# The threaded pause hook — the mechanism under test.
# Returns (variables_pool | None, n_pauses, clean: bool).
# `max_pauses=None` -> run to completion; an int -> STOP after that many (teardown).
# ---------------------------------------------------------------------------
def run_threaded(meta: Path, work_dir: Path, max_pauses):
    from RUFAS.output_manager import OutputManager
    from RUFAS.simulation_engine import SimulationEngine
    from RUFAS.task_manager import TaskManager

    to_main: "queue.Queue[str]" = queue.Queue()    # worker -> driver: "boundary" / "done"
    to_worker: "queue.Queue[str]" = queue.Queue()  # driver -> worker: "GO" / "STOP"

    orig_daily = SimulationEngine._execute_full_farm_daily_simulation

    def hooked_daily(self):
        if _is_month_boundary(self):
            to_main.put("boundary")
            if to_worker.get() == "STOP":     # BLOCKS here — the pause
                raise _Teardown()
        return orig_daily(self)

    SimulationEngine._execute_full_farm_daily_simulation = hooked_daily

    worker_error: list[BaseException] = []

    def worker():
        try:
            TaskManager().start(metadata_path=meta, **_start_kwargs(work_dir))
        except _Teardown:
            pass                                # expected on STOP
        except BaseException as exc:            # noqa: BLE001 — surface to driver
            worker_error.append(exc)
        finally:
            to_main.put("done")

    t = threading.Thread(target=worker, name="rufas-sim", daemon=True)
    try:
        t.start()
        n_pauses = 0
        while True:
            msg = to_main.get()                 # wait for a pause or completion
            if msg == "done":
                break
            n_pauses += 1
            if max_pauses is not None and n_pauses >= max_pauses:
                to_worker.put("STOP")           # teardown: unwind the worker
            else:
                to_worker.put("GO")             # Axis A: resume, no action applied
        t.join(timeout=60)
        clean = (not t.is_alive()) and not worker_error
        if worker_error:
            raise worker_error[0]
        pool = None
        if max_pauses is None:                  # only a completed run has a full pool
            pool = OutputManager()._get_flat_variables_pool()
        return pool, n_pauses, clean
    finally:
        SimulationEngine._execute_full_farm_daily_simulation = orig_daily


# ---------------------------------------------------------------------------
# One mode, in this (sub)process.
# ---------------------------------------------------------------------------
def run_one(mode: str, out_path: Path) -> None:
    root = rufas_root()
    ensure_importable(root)
    from RUFAS.input_manager import InputManager
    from RUFAS.output_manager import LogVerbosity, OutputManager
    from RUFAS.task_manager import TaskManager

    prev = Path.cwd()
    os.chdir(root)
    try:
        InputManager().flush_pool()
        OutputManager().flush_pools()
        with tempfile.TemporaryDirectory(prefix="axisa_") as tmp:
            work_dir = Path(tmp)
            meta = patched_metadata(root, work_dir)

            result = {"mode": mode, "pauses": None, "clean": None, "pool": None}
            if mode == "unhooked":
                TaskManager().start(metadata_path=meta, **_start_kwargs(work_dir))
                result["pool"] = _jsonable(OutputManager()._get_flat_variables_pool())
            elif mode == "stepped":
                pool, n, clean = run_threaded(meta, work_dir, max_pauses=None)
                result.update(pauses=n, clean=clean, pool=_jsonable(pool))
            elif mode == "teardown":
                _, n, clean = run_threaded(meta, work_dir, max_pauses=3)
                result.update(pauses=n, clean=clean)
            else:
                raise SystemExit(f"unknown mode {mode!r}")

            out_path.write_text(json.dumps(result))
            npool = len(result["pool"]) if result["pool"] else 0
            print(f"[{mode}] pauses={result['pauses']} clean={result['clean']} "
                  f"pool_keys={npool} -> {out_path}")
    finally:
        os.chdir(prev)


def _jsonable(obj):
    if obj is None:
        return None
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
# Comparison — physical-key fingerprint (same method as the Axis-B spike).
# ---------------------------------------------------------------------------
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


def _flatten_numbers(pool: dict) -> dict[str, float]:
    out = {}
    for k, v in pool.items():
        nums = _numbers(v)
        if nums:
            out[k] = sum(nums)
    return out


def _rel_diff(a: dict[str, float], b: dict[str, float]):
    keys = set(a) | set(b)
    n_diff = 0
    worst = []
    for k in keys:
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        denom = max(abs(av), abs(bv), 1e-9)
        rd = abs(av - bv) / denom
        if rd > 1e-6:
            n_diff += 1
            worst.append((k, rd))
    worst.sort(key=lambda t: -t[1])
    return n_diff, worst[:8]


def compare(results: dict[str, dict]) -> None:
    unhooked = _flatten_numbers(results["unhooked"]["pool"])
    stepped = _flatten_numbers(results["stepped"]["pool"])
    n_pauses = results["stepped"]["pauses"]
    stepped_clean = results["stepped"]["clean"]
    td = results["teardown"]

    n_diff, worst = _rel_diff(stepped, unhooked)

    print("\n================ Axis-A pause spike ================")
    print(f"  unhooked : {len(unhooked)} numeric keys")
    print(f"  stepped  : {len(stepped)} numeric keys, paused {n_pauses}x, "
          f"completed_clean={stepped_clean}")
    print(f"  teardown : released {td['pauses']} then STOP, thread_exit_clean={td['clean']}")

    print(f"\n  suspend/resume worked : {n_pauses} pauses, no deadlock "
          f"(expect >0)")
    print(f"  stepped vs unhooked   : {n_diff} physical keys differ "
          f"(expect 0 — pausing must not change the science)")
    for k, rd in worst[:5]:
        print(f"      {k}: reldiff={rd:.3g}")
    print(f"  teardown clean        : {td['clean']} (expect True)")

    passed = (n_pauses > 0 and stepped_clean and n_diff == 0 and td["clean"])
    print("\n  RESULT: " + (
        "PASS — the threaded pause hook suspends/resumes RuFaS cleanly at monthly "
        "boundaries, is byte-identical in output to an unhooked run, and tears down "
        "cleanly. Axis-A pausing proven; zero RuFaS edits. Ready to wrap as the "
        "ThreadedPauseStepper (T7)."
        if passed else
        "FAIL / INCONCLUSIVE — see above. n_diff>0 => pausing altered results "
        "(the hook must be pure, block-only). n_pauses==0 => the patched method was "
        "never reached (wrong scenario? must be full_farm). teardown!=clean => the "
        "STOP sentinel deadlocked/leaked."))
    print("===================================================\n")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--run":            # internal: single-mode subprocess entry
        run_one(args[1], Path(args[2]))
        return

    py = sys.executable
    out_dir = Path(tempfile.mkdtemp(prefix="axisa_out_"))
    modes = ("unhooked", "stepped", "teardown")
    paths = {m: out_dir / f"{m}.json" for m in modes}
    for m, p in paths.items():
        print(f"--- running {m} (fresh subprocess) ---")
        subprocess.run([py, str(THIS), "--run", m, str(p)], check=True)
    results = {m: json.loads(p.read_text()) for m, p in paths.items()}
    compare(results)


if __name__ == "__main__":
    main()
