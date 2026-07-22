#!/usr/bin/env python3
"""Equivalence check for the SHIPPED field appliers (fertilizer, manure).

Unlike `spike_axisb_fertilizer.py` (which reimplements the mechanism inline), this drives
the actual `rufas_rl.appliers` classes the env uses, so it verifies the code that ships.
For each lever: applying a per-field multiplier at the first pause must equal injecting the
same scaled schedule at construction (0 physical diffs), and must differ from an untouched
run (the change bites).

    python scripts/verify_field_appliers.py            # both levers
    python scripts/verify_field_appliers.py fertilizer
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import threading
import queue
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from rufas_rl.appliers import FertilizerApplier, ManureApplier, set_engine  # noqa: E402
from rufas_rl.bootstrap import rufas_cwd  # noqa: E402
from rufas_rl.runtime import flush_singletons, prepare_metadata, start_kwargs  # noqa: E402

MULT = 1.5
SPEC_KEY = {"fertilizer": "fertilizer_management_specification",
            "manure": "manure_management_specification"}


def _applier(lever):
    return FertilizerApplier() if lever == "fertilizer" else ManureApplier()


def _scaled_patch(lever):
    """Construction-time input_patch scaling every field's schedule by MULT."""
    from RUFAS.input_manager import InputManager

    pool = InputManager()._InputManager__pool
    patch = {}
    for fname in [k for k in pool if k.startswith("field_")]:
        schedule = pool[fname][SPEC_KEY[lever]]
        base = pool[schedule]
        frag = {}
        for key in ("nitrogen_masses", "phosphorus_masses"):
            if key in base:
                frag[key] = [v * MULT for v in base[key]]
        patch[schedule] = frag
    return patch


def run_threaded(meta, work, lever, mutate):
    from RUFAS.output_manager import OutputManager
    from RUFAS.simulation_engine import SimulationEngine
    from RUFAS.task_manager import TaskManager

    to_main: "queue.Queue" = queue.Queue()
    to_worker: "queue.Queue[str]" = queue.Queue()
    orig = SimulationEngine._execute_full_farm_daily_simulation
    applier = _applier(lever)

    def hooked(self):
        if self.time.current_date.day == 1:
            to_main.put(self)
            to_worker.get()
        return orig(self)

    SimulationEngine._execute_full_farm_daily_simulation = hooked
    err: list = []

    def worker():
        try:
            TaskManager().start(metadata_path=meta, **start_kwargs(work))
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            to_main.put("done")

    t = threading.Thread(target=worker, daemon=True)
    try:
        t.start()
        n = 0
        while True:
            msg = to_main.get()
            if msg == "done":
                break
            n += 1
            if mutate and n == 1:
                set_engine(msg)
                applier.apply([MULT] * len(msg.field_manager.fields))
            to_worker.put("GO")
        t.join(timeout=60)
        if err:
            raise err[0]
        return OutputManager()._get_flat_variables_pool()
    finally:
        SimulationEngine._execute_full_farm_daily_simulation = orig


def run_construct(meta, work, lever):
    from RUFAS.output_manager import OutputManager
    from RUFAS.task_manager import TaskManager

    orig = TaskManager._handle_simulation_engine_run_tasks

    def handler(args, im, om, tid, pg, flush):
        args["input_patch"] = _scaled_patch(lever)
        return orig(args, im, om, tid, pg, flush)

    TaskManager._handle_simulation_engine_run_tasks = staticmethod(handler)
    try:
        TaskManager().start(metadata_path=meta, **start_kwargs(work))
        return OutputManager()._get_flat_variables_pool()
    finally:
        TaskManager._handle_simulation_engine_run_tasks = orig


def run_one(lever, mode, out_path):
    with rufas_cwd():
        flush_singletons()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            meta = prepare_metadata("input/task_manager_metadata.json", work, seed=42)
            if mode == "construct":
                pool = run_construct(meta, work, lever)
            else:
                pool = run_threaded(meta, work, lever, mutate=(mode == "act"))
            out_path.write_text(json.dumps({"pool": _jsonable(pool)}))
            print(f"[{lever}/{mode}] keys={len(pool)} -> {out_path.name}")


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return {k: (v if _ok(v) else str(v)) for k, v in o.items()}


def _ok(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


def _numbers(v):
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        return [n for x in v for n in _numbers(x)]
    if isinstance(v, dict):
        return [n for x in v.values() for n in _numbers(x)]
    return []


def _flatten(pool):
    out = {}
    for k, v in pool.items():
        if "setup_fertilizer" in k or "setup_manure" in k or "Schedule" in k:
            continue
        nums = _numbers(v)
        if nums:
            out[k] = sum(nums)
    return out


def _diff(a, b):
    keys = set(a) | set(b)
    return sum(1 for k in keys
               if abs(a.get(k, 0.0) - b.get(k, 0.0)) / max(abs(a.get(k, 0.0)), abs(b.get(k, 0.0)), 1e-9) > 1e-6)


def verify(lever):
    py = sys.executable
    out = Path(tempfile.mkdtemp())
    pools = {}
    for mode in ("noop", "act", "construct"):
        p = out / f"{lever}_{mode}.json"
        subprocess.run([py, str(THIS), "--run", lever, mode, str(p)], check=True)
        pools[mode] = _flatten(json.loads(p.read_text())["pool"])
    bites = _diff(pools["act"], pools["noop"])
    equal = _diff(pools["act"], pools["construct"])
    ok = bites > 0 and equal == 0
    print(f"\n  {lever}: act vs noop = {bites} differ (want >0); "
          f"act vs construct = {equal} differ (want 0)  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = sys.argv[1:]
    if args and args[0] == "--run":
        run_one(args[1], args[2], Path(args[3]))
        return
    levers = [args[0]] if args else ["fertilizer", "manure"]
    results = {lv: verify(lv) for lv in levers}
    print("\n" + ("ALL PASS" if all(results.values()) else "SOME FAILED"))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
