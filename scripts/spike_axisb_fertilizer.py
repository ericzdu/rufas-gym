#!/usr/bin/env python3
"""Axis-B FERTILIZER applier spike — the field-lever go/no-go.

The ration lever is proven (spike_axisb_rations / spike_pause_and_mutate). This spike
asks the same question for the first FIELD lever, fertilizer, because the field levers are
the ones that carry the cross-year story the project is really after (soil-N carryover,
legume-then-corn). If this passes, per-period fertilizer control is viable; if it fails,
fertilizer can only be set once per episode at construction.

The mechanism mirrors the ration applier, but fertilizer is applied by re-invoking RuFaS's
OWN construction-time setup function against the live field:

  1. mutate the InputManager pool's fertilizer schedule (scale nitrogen),
  2. call FieldManager._setup_fertilizer_events(schedule_name) — which re-reads the pool
     and rebuilds the events exactly as construction would,
  3. reassign the returned (mixes, events) onto the live Field.

Why applying at the first pause equals construction injection: RuFaS's fertilizer events
fire on specific dates (here 2013 day 126 and 2017 day 126). The first pause is 2013-01-01,
before any of them, and Field._filter_events (field.py:1232) drops past-dated events and
keeps only future ones each day — so replacing the whole list before the first event fires
is equivalent to having configured it that way from the start.

Three modes, each a FRESH SUBPROCESS dumping variables_pool:

  noop      — threaded pause, release every boundary, no change. Control.
  act       — at the first pause, live-mutate the fertilizer schedule (scale N x1.5) and
              rebuild events on the live fields, then resume.
  construct — the SAME scaled fertilizer injected at CONSTRUCTION via input_patch (oracle).

PASS: act DIFFERS from noop (the change bites) AND act == construct (applying at the pause
== construction injection).

Usage:
  ../RuFaS/venv/bin/python scripts/spike_axisb_fertilizer.py
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
sys.path.insert(0, str(THIS.parent.parent))

from rufas_rl.bootstrap import rufas_cwd  # noqa: E402
from rufas_rl.runtime import prepare_metadata, flush_singletons, start_kwargs  # noqa: E402

N_SCALE = 1.5  # scale nitrogen masses by this
# field index -> its fertilizer schedule blob name (from the freestall scenario).
FIELD_FERTILIZER = {0: "fertilizer_schedule_1", 1: "fertilizer_schedule_2"}


def _is_month_boundary(engine) -> bool:
    return engine.time.current_date.day == 1


def scale_fertilizer(blob: dict) -> dict:
    """Return a copy of a fertilizer schedule with nitrogen masses scaled."""
    mod = copy.deepcopy(blob)
    mod["nitrogen_masses"] = [n * N_SCALE for n in blob["nitrogen_masses"]]
    return mod


# --- the applier: called on the DRIVER thread while the worker is blocked -----
def apply_fertilizer_mutation(engine) -> None:
    from RUFAS.input_manager import InputManager
    from RUFAS.biophysical.field.manager.field_manager import FieldManager

    im = InputManager()
    pool = im._InputManager__pool  # name-mangled private pool
    for idx, field in enumerate(engine.field_manager.fields):
        schedule_name = FIELD_FERTILIZER[idx]
        # 1. mutate the pool the setup function will re-read
        pool[schedule_name] = scale_fertilizer(pool[schedule_name])
        # 2. re-invoke RuFaS's own setup fn on the modified config
        mixes, events = FieldManager._setup_fertilizer_events(schedule_name)
        # 3. reassign both returned pieces onto the live field
        field.available_fertilizer_mixes = mixes
        field.fertilizer_events = events


def run_threaded(meta: Path, work_dir: Path, mutate: bool):
    from RUFAS.output_manager import OutputManager
    from RUFAS.simulation_engine import SimulationEngine
    from RUFAS.task_manager import TaskManager

    to_main: "queue.Queue" = queue.Queue()
    to_worker: "queue.Queue[str]" = queue.Queue()
    orig = SimulationEngine._execute_full_farm_daily_simulation

    def hooked(self):
        if _is_month_boundary(self):
            to_main.put(self)
            to_worker.get()
        return orig(self)

    SimulationEngine._execute_full_farm_daily_simulation = hooked
    worker_error: list[BaseException] = []

    def worker():
        try:
            TaskManager().start(metadata_path=meta, **start_kwargs(work_dir))
        except BaseException as exc:  # noqa: BLE001
            worker_error.append(exc)
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
                apply_fertilizer_mutation(msg)  # msg is the live engine
            to_worker.put("GO")
        t.join(timeout=60)
        if worker_error:
            raise worker_error[0]
        return OutputManager()._get_flat_variables_pool(), n
    finally:
        SimulationEngine._execute_full_farm_daily_simulation = orig


def run_construct(meta: Path, work_dir: Path):
    from RUFAS.output_manager import OutputManager
    from RUFAS.task_manager import TaskManager

    orig_handler = TaskManager._handle_simulation_engine_run_tasks

    def handler(args, im, om, task_id, produce_graphics, should_flush):
        patch = {}
        for schedule_name in FIELD_FERTILIZER.values():
            blob = im.get_data(schedule_name)
            patch[schedule_name] = {"nitrogen_masses": scale_fertilizer(blob)["nitrogen_masses"]}
        args["input_patch"] = patch
        return orig_handler(args, im, om, task_id, produce_graphics, should_flush)

    TaskManager._handle_simulation_engine_run_tasks = staticmethod(handler)
    try:
        TaskManager().start(metadata_path=meta, **start_kwargs(work_dir))
        return OutputManager()._get_flat_variables_pool()
    finally:
        TaskManager._handle_simulation_engine_run_tasks = orig_handler


def run_one(mode: str, out_path: Path) -> None:
    with rufas_cwd():
        flush_singletons()
        with tempfile.TemporaryDirectory(prefix="fert_") as tmp:
            work = Path(tmp)
            meta = prepare_metadata("input/task_manager_metadata.json", work, seed=42)
            if mode == "noop":
                pool, n = run_threaded(meta, work, mutate=False)
            elif mode == "act":
                pool, n = run_threaded(meta, work, mutate=True)
            elif mode == "construct":
                pool, n = run_construct(meta, work), None
            else:
                raise SystemExit(f"unknown mode {mode!r}")
            out_path.write_text(json.dumps({"mode": mode, "pauses": n, "pool": _jsonable(pool)}))
            print(f"[{mode}] pauses={n} keys={len(pool) if pool else 0} -> {out_path}")


def _jsonable(obj):
    if obj is None:
        return None
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return {k: _coerce(v) for k, v in obj.items()}


def _coerce(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


# --- physical-key fingerprint comparison (same as the other spikes) ----------
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
        if "setup_fertilizer" in k or "FertilizerSchedule" in k:
            continue  # instrumentation of the setter call itself, not a physical output
        nums = _numbers(v)
        if nums:
            out[k] = sum(nums)
    return out


def _rel_diff(a, b):
    keys = set(a) | set(b)
    n = 0
    worst = []
    for k in keys:
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        rd = abs(av - bv) / max(abs(av), abs(bv), 1e-9)
        if rd > 1e-6:
            n += 1
            worst.append((k, rd))
    worst.sort(key=lambda kv: -kv[1])
    return n, worst[:8]


def compare(results):
    noop = _flatten(results["noop"]["pool"])
    act = _flatten(results["act"]["pool"])
    con = _flatten(results["construct"]["pool"])
    an_n, an_w = _rel_diff(act, noop)
    ac_n, ac_w = _rel_diff(act, con)

    print("\n========== Axis-B fertilizer applier spike ==========")
    print(f"  noop      : {len(noop)} keys, paused {results['noop']['pauses']}x")
    print(f"  act       : {len(act)} keys, paused {results['act']['pauses']}x, "
          f"fertilizer rebuilt at pause #1")
    print(f"  construct : {len(con)} keys (oracle)")
    print(f"\n  act vs noop      : {an_n} keys differ  (expect >0 — scaling N bites)")
    for k, rd in an_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")
    print(f"  act vs construct : {ac_n} keys differ  (expect 0 — applier == construction)")
    for k, rd in ac_w[:5]:
        print(f"      {k}: reldiff={rd:.3g}")

    passed = an_n > 0 and ac_n == 0
    print("\n  RESULT: " + (
        "PASS — rebuilding fertilizer events on the live field at a pause is physically "
        "identical to construction injection. The field-op applier mechanism works; "
        "fertilizer can be a per-period lever."
        if passed else
        "FAIL / INCONCLUSIVE — act==noop => the change didn't take; act!=construct => "
        "live rebuild is NOT equivalent (timing, or events already fired, or missing "
        "reassignment target)."))
    print("=====================================================\n")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--run":
        run_one(args[1], Path(args[2]))
        return
    py = sys.executable
    out_dir = Path(tempfile.mkdtemp(prefix="fert_out_"))
    modes = ("noop", "act", "construct")
    paths = {m: out_dir / f"{m}.json" for m in modes}
    for m, p in paths.items():
        print(f"--- running {m} (fresh subprocess) ---")
        subprocess.run([py, str(THIS), "--run", m, str(p)], check=True)
    results = {m: json.loads(p.read_text()) for m, p in paths.items()}
    compare(results)


if __name__ == "__main__":
    main()
