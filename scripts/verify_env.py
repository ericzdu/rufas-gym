#!/usr/bin/env python3
"""Verification for the packaged env — the two properties everything else rests on.

**equivalence (V2/V10)** — a simulation stepped through `ThreadedPauseStepper` must
produce exactly the same numbers as one run straight through with no hook at all. If this
fails, pausing is perturbing the science and every result from the env is void. The
original Axis-A spike proved this for a hook on one method; the packaged stepper hooks all
four dispatchable daily methods, so the property is re-proven here for the code that
actually ships.

**pauses** — every scenario must actually stop. This is the failure mode worth fearing:
hooking only the full_farm method meant a `field_only` scenario ran to completion, looked
completely normal, and silently never paused. A false pass. This check asserts a nonzero
pause count per scenario, so that cannot go unnoticed again.

Each run is a fresh subprocess, so the known ~5x in-process slowdown cannot confound the
comparison.

    python scripts/verify_env.py                 # both checks, default scenario + field_only
    python scripts/verify_env.py equivalence
    python scripts/verify_env.py pauses
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from rufas_rl.bootstrap import rufas_cwd  # noqa: E402
from rufas_rl.runtime import flush_singletons, prepare_metadata, start_kwargs  # noqa: E402
from rufas_rl.spec import load_spec  # noqa: E402
from rufas_rl.stepper import ThreadedPauseStepper  # noqa: E402

DEFAULT_SCENARIO = "input/task_manager_metadata.json"
FIELD_ONLY_SCENARIO = "input/kimberly_rotation_task_manager_metadata.json"

TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# fingerprinting: reduce the variables_pool to one number per key
# --------------------------------------------------------------------------
def _numbers(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    return []


def fingerprint(pool: dict) -> dict[str, float]:
    out = {}
    for key, value in pool.items():
        nums = _numbers(value)
        if nums:
            out[key] = sum(nums)
    return out


def compare(a: dict[str, float], b: dict[str, float]) -> tuple[int, list[tuple[str, float]]]:
    worst: list[tuple[str, float]] = []
    for key in set(a) | set(b):
        av, bv = a.get(key, 0.0), b.get(key, 0.0)
        rel = abs(av - bv) / max(abs(av), abs(bv), 1e-9)
        if rel > TOLERANCE:
            worst.append((key, rel))
    worst.sort(key=lambda kv: -kv[1])
    return len(worst), worst[:8]


# --------------------------------------------------------------------------
# the two run modes, each executed in its own subprocess
# --------------------------------------------------------------------------
def run_unhooked(scenario: str) -> dict:
    from RUFAS.output_manager import OutputManager
    from RUFAS.task_manager import TaskManager

    with tempfile.TemporaryDirectory(prefix="verify_plain_") as tmp:
        work = Path(tmp)
        metadata = prepare_metadata(scenario, work, seed=42)
        TaskManager().start(metadata_path=metadata, **start_kwargs(work))
        return fingerprint(OutputManager()._get_flat_variables_pool())


def run_stepped(scenario: str, cadence: str = "monthly") -> tuple[dict, int]:
    with tempfile.TemporaryDirectory(prefix="verify_step_") as tmp:
        work = Path(tmp)
        metadata = prepare_metadata(scenario, work, seed=42)
        stepper = ThreadedPauseStepper(metadata, work, cadence=cadence)
        try:
            engine = stepper.start()
            while engine is not None:
                engine = stepper.advance()  # resume without applying any action
            return fingerprint(stepper.variables_pool()), stepper.n_pauses
        finally:
            stepper.close()


def _child(mode: str, scenario: str, out_path: Path) -> None:
    with rufas_cwd():
        flush_singletons()
        if mode == "unhooked":
            result = {"pool": run_unhooked(scenario), "pauses": 0}
        elif mode == "stepped":
            pool, pauses = run_stepped(scenario)
            result = {"pool": pool, "pauses": pauses}
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    out_path.write_text(json.dumps(result))
    print(f"  [{mode}] keys={len(result['pool'])} pauses={result['pauses']}")


def _spawn(mode: str, scenario: str, out_dir: Path) -> dict:
    out_path = out_dir / f"{mode}.json"
    subprocess.run(
        [sys.executable, str(THIS), "--child", mode, scenario, str(out_path)],
        check=True,
    )
    return json.loads(out_path.read_text())


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_equivalence(scenario: str = DEFAULT_SCENARIO) -> bool:
    print(f"\n=== equivalence: stepped vs unhooked ({scenario}) ===")
    out_dir = Path(tempfile.mkdtemp(prefix="verify_out_"))
    unhooked = _spawn("unhooked", scenario, out_dir)
    stepped = _spawn("stepped", scenario, out_dir)

    n_diff, worst = compare(stepped["pool"], unhooked["pool"])
    print(f"  paused {stepped['pauses']}x")
    print(f"  {n_diff} of {len(unhooked['pool'])} physical keys differ (expect 0)")
    for key, rel in worst:
        print(f"      {key}: reldiff={rel:.3g}")

    ok = n_diff == 0 and stepped["pauses"] > 0
    print("  RESULT:", "PASS — pausing does not change the science" if ok else "FAIL")
    return ok


def check_pauses(scenarios: list[str]) -> bool:
    print("\n=== pauses: every simulation_type must actually stop ===")
    all_ok = True
    for scenario in scenarios:
        spec = load_spec(scenario)
        out_dir = Path(tempfile.mkdtemp(prefix="verify_pause_"))
        try:
            result = _spawn("stepped", scenario, out_dir)
            pauses = result["pauses"]
            expected = spec.n_boundaries("monthly")
            ok = pauses > 0
            print(
                f"  {spec.simulation_type:>14}  paused {pauses:>4}x "
                f"(scenario spans {expected} months)  "
                + ("OK" if ok else "FAIL — hook never fired")
            )
        except subprocess.CalledProcessError:
            ok = False
            print(f"  {spec.simulation_type:>14}  FAILED to run")
        all_ok &= ok
    print("  RESULT:", "PASS — all scenario types pause" if all_ok else "FAIL")
    return all_ok


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--child":
        _child(args[1], args[2], Path(args[3]))
        return

    which = args[0] if args else "all"
    results = []
    if which in ("all", "equivalence"):
        results.append(check_equivalence())
    if which in ("all", "pauses"):
        results.append(check_pauses([DEFAULT_SCENARIO, FIELD_ONLY_SCENARIO]))

    print("\n" + ("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
