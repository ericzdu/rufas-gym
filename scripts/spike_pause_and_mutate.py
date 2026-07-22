#!/usr/bin/env python3
"""
Axis-A + Axis-B integration spike — CHANGE A VALUE WHILE SUSPENDED.

The Axis-A spike proved RuFaS pauses/resumes cleanly (scripts/spike_axisa_pause.py).
The Axis-B spike proved a live ration mutation == construction injection, but it
triggered the mutation from inside `formulate_rations`, not from a pause
(scripts/spike_axisb_rations.py). This spike closes the gap: it applies the change
*while the simulation is genuinely suspended at a pause boundary* — exactly what a
`step(action)` will do — and proves the change both (a) takes effect and (b) is
identical to the proven construction-injection path.

Mechanism: run RuFaS on a worker thread behind the zero-edit threaded pause hook
(monkey-patched `SimulationEngine._execute_full_farm_daily_simulation`, blocking on a
hand-off queue at monthly boundaries). At the FIRST pause the DRIVER thread — with the
worker provably blocked (no race) — live-mutates `RationManager` (the proven Axis-B
applier), then releases the worker. The first pause is 2013-01-01, which lands BEFORE
day-1's `_execute_ration_planning` (`next_ration_reformulation` is initialised to the
start date, simulation_engine.py:246), so the mutation is seen by the first
formulation — the same point construction injection takes effect.

Three modes, each in a FRESH SUBPROCESS, each dumping variables_pool:

  noop      — threaded pause, release every boundary, NO mutation. Control
              (== unhooked, already shown by the Axis-A spike).
  act       — threaded pause; at the first pause the driver live-mutates
              RationManager to the reversed ration, then resumes to completion.
  construct — the SAME reversed ration injected at CONSTRUCTION via input_patch
              (the proven-correct oracle).

PASS (changing a value while suspended works AND is correct):
  * act DIFFERS from noop       (the mid-pause change actually bites), AND
  * act == construct            (applying at the pause == construction injection).

Usage:
  driver:  ../RuFaS/venv/bin/python scripts/spike_pause_and_mutate.py
"""

from __future__ import annotations

import copy
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
    """Reverse each group's ration_percentage vector (sum-preserving, feasible).
    Identical transform used by `act` (at the pause) and `construct` (at build)."""
    mod = copy.deepcopy(feed_blob)
    for ration in mod["rations"]:
        pcts = [f["ration_percentage"] for f in ration["feeds"]]
        for f, p in zip(ration["feeds"], reversed(pcts)):
            f["ration_percentage"] = p
    return mod


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


def _is_month_boundary(engine) -> bool:
    return engine.time.current_date.day == 1


# The Axis-B applier, called from the DRIVER thread while the worker is blocked.
def _apply_ration_mutation() -> None:
    from RUFAS.input_manager import InputManager
    from RUFAS.biophysical.animal.ration.ration_manager import RationManager
    feed = InputManager().get_data("feed")
    mod = reverse_rations(feed)
    RationManager.set_user_defined_rations(mod)
    RationManager.set_user_defined_ration_tolerance(mod)


def run_threaded_with_action(meta: Path, work_dir: Path, mutate: bool):
    """Threaded pause hook. If `mutate`, apply the ration change at the FIRST pause
    (driver thread, worker blocked). Returns (variables_pool, n_pauses, clean)."""
    from RUFAS.output_manager import OutputManager
    from RUFAS.simulation_engine import SimulationEngine
    from RUFAS.task_manager import TaskManager

    to_main: "queue.Queue[str]" = queue.Queue()
    to_worker: "queue.Queue[str]" = queue.Queue()
    orig_daily = SimulationEngine._execute_full_farm_daily_simulation

    def hooked_daily(self):
        if _is_month_boundary(self):
            to_main.put("boundary")
            to_worker.get()                    # BLOCKS — the pause
        return orig_daily(self)

    SimulationEngine._execute_full_farm_daily_simulation = hooked_daily
    worker_error: list[BaseException] = []

    def worker():
        try:
            TaskManager().start(metadata_path=meta, **_start_kwargs(work_dir))
        except BaseException as exc:           # noqa: BLE001
            worker_error.append(exc)
        finally:
            to_main.put("done")

    t = threading.Thread(target=worker, name="rufas-sim", daemon=True)
    try:
        t.start()
        n_pauses = 0
        while True:
            if to_main.get() == "done":
                break
            n_pauses += 1
            if mutate and n_pauses == 1:       # change a value WHILE SUSPENDED
                _apply_ration_mutation()
            to_worker.put("GO")                # resume
        t.join(timeout=60)
        clean = (not t.is_alive()) and not worker_error
        if worker_error:
            raise worker_error[0]
        return OutputManager()._get_flat_variables_pool(), n_pauses, clean
    finally:
        SimulationEngine._execute_full_farm_daily_simulation = orig_daily


def run_construct(meta: Path, work_dir: Path):
    """Oracle: reversed ration injected at construction via input_patch."""
    from RUFAS.output_manager import OutputManager
    from RUFAS.task_manager import TaskManager

    orig_handler = TaskManager._handle_simulation_engine_run_tasks

    def handler(args, input_manager, output_manager, task_id,
                produce_graphics, should_flush_im_pool):
        feed = input_manager.get_data("feed")
        mod = reverse_rations(feed)
        args["input_patch"] = {"feed": {"rations": mod["rations"]}}
        return orig_handler(args, input_manager, output_manager, task_id,
                            produce_graphics, should_flush_im_pool)

    TaskManager._handle_simulation_engine_run_tasks = staticmethod(handler)
    try:
        TaskManager().start(metadata_path=meta, **_start_kwargs(work_dir))
        return OutputManager()._get_flat_variables_pool()
    finally:
        TaskManager._handle_simulation_engine_run_tasks = orig_handler


def run_one(mode: str, out_path: Path) -> None:
    root = rufas_root()
    ensure_importable(root)
    from RUFAS.input_manager import InputManager
    from RUFAS.output_manager import OutputManager

    prev = Path.cwd()
    os.chdir(root)
    try:
        InputManager().flush_pool()
        OutputManager().flush_pools()
        with tempfile.TemporaryDirectory(prefix="pam_") as tmp:
            work_dir = Path(tmp)
            meta = patched_metadata(root, work_dir)
            result = {"mode": mode, "pauses": None, "clean": None, "pool": None}
            if mode == "noop":
                pool, n, clean = run_threaded_with_action(meta, work_dir, mutate=False)
                result.update(pauses=n, clean=clean, pool=_jsonable(pool))
            elif mode == "act":
                pool, n, clean = run_threaded_with_action(meta, work_dir, mutate=True)
                result.update(pauses=n, clean=clean, pool=_jsonable(pool))
            elif mode == "construct":
                result["pool"] = _jsonable(run_construct(meta, work_dir))
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


# ---- physical-key fingerprint comparison (same as the other spikes) ----
# These keys log the INPUT to the ration setter (what it was called with), not a
# simulation RESULT; they differ trivially because `act` calls the setter (once,
# reversed) while `construct` routes the same ration through input_patch. Exclude
# them so the comparison is over physical outputs only.
INSTRUMENTATION = ("set_user_defined_rations", "set_ration_feeds",
                   "set_user_defined_ration_tolerance")


def _is_instrumentation(key: str) -> bool:
    return any(tag in key for tag in INSTRUMENTATION)


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
        if _is_instrumentation(k):
            continue
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
    noop = _flatten_numbers(results["noop"]["pool"])
    act = _flatten_numbers(results["act"]["pool"])
    construct = _flatten_numbers(results["construct"]["pool"])

    an_n, an_w = _rel_diff(act, noop)
    ac_n, ac_w = _rel_diff(act, construct)

    print("\n============ pause-and-mutate spike (Axis A + Axis B) ============")
    print(f"  noop      : {len(noop)} keys, paused {results['noop']['pauses']}x")
    print(f"  act       : {len(act)} keys, paused {results['act']['pauses']}x, "
          f"mutated at pause #1")
    print(f"  construct : {len(construct)} keys (oracle)")
    print(f"\n  act vs noop      : {an_n} keys differ  (expect >0 — "
          f"changing a value while suspended bites)")
    for k, rd in an_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")
    print(f"  act vs construct : {ac_n} keys differ  (expect 0 — "
          f"pause-time mutation == construction injection)")
    for k, rd in ac_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")

    passed = an_n > 0 and ac_n == 0
    print("\n  RESULT: " + (
        "PASS — a value changed WHILE THE SIM IS SUSPENDED takes effect on resume "
        "and is physically identical to injecting it at construction. Axis A + Axis B "
        "integrate; `step(action)` semantics are sound."
        if passed else
        "FAIL / INCONCLUSIVE — see diffs. act==noop => the mutation didn't take "
        "(applied too late / wrong pause). act!=construct => pause-time application "
        "is NOT equivalent (timing vs the first formulation)."))
    print("=================================================================\n")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--run":
        run_one(args[1], Path(args[2]))
        return
    py = sys.executable
    out_dir = Path(tempfile.mkdtemp(prefix="pam_out_"))
    modes = ("noop", "act", "construct")
    paths = {m: out_dir / f"{m}.json" for m in modes}
    for m, p in paths.items():
        print(f"--- running {m} (fresh subprocess) ---")
        subprocess.run([py, str(THIS), "--run", m, str(p)], check=True)
    results = {m: json.loads(p.read_text()) for m, p in paths.items()}
    compare(results)


if __name__ == "__main__":
    main()
