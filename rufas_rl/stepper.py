"""Axis A — pausing RuFaS mid-run without editing it.

RuFaS runs start-to-finish inside one blocking `TaskManager.start` call. To step it we
run that call on a worker thread and monkeypatch the engine's daily-step method so that,
at each decision boundary, the worker blocks on a queue until the driver releases it.

The important property: **the paused thread's own call stack is the saved farm state.**
Nothing is serialized, nothing is reconstructed, and the simulation resumes in exactly
the state it stopped in. This is why the mechanism is dynamics-preserving — the hook is
pure control flow, it never touches a number.

Only one side ever runs at a time (a strict handoff), so the GIL and RuFaS's module-level
globals are not a concern.

Verified by `scripts/spike_axisa_pause.py`: 84 pauses on the 7-year freestall scenario,
resumed to completion with no deadlock, and a `variables_pool` identical to an unhooked
run (0 of 2702 physical keys differ).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable

# Every daily-step method RuFaS can dispatch to, from
# `SimulationEngine._simulation_type_to_daily_simulation_function`. We patch all four
# rather than just the full_farm one: a scenario's `simulation_type` decides which is
# called, and hooking only one means a field_only or animals_only scenario runs to
# completion and *silently never pauses* — a false pass that looks like success.
DAILY_METHODS = (
    "_execute_full_farm_daily_simulation",
    "_execute_field_and_feed_daily_simulation",
    "_execute_field_only_simulation",
    "_execute_animals_only_daily_simulation",
)

BoundaryPredicate = Callable[[object], bool]


def monthly(engine) -> bool:
    """First day of each month. Aligns with the default 30-day ration formulation."""
    return engine.time.current_date.day == 1


def yearly(engine) -> bool:
    """First day of January — the cadence at which field operations can change."""
    return engine.time.current_date.day == 1 and engine.time.current_date.month == 1


def daily(engine) -> bool:
    return True


CADENCES: dict[str, BoundaryPredicate] = {
    "monthly": monthly,
    "yearly": yearly,
    "daily": daily,
}


def get_cadence(name: str) -> BoundaryPredicate:
    try:
        return CADENCES[name]
    except KeyError:
        raise ValueError(f"Unknown cadence {name!r}; expected one of {sorted(CADENCES)}") from None


class _Teardown(BaseException):
    """Raised inside the hook to unwind the worker thread on `close()`.

    Deliberately a `BaseException`, not an `Exception`. Abandoning an episode early —
    which `env.reset()` and `max_steps` do constantly — unwinds through RuFaS's own call
    stack, and `TaskManager` wraps task execution in a broad `except Exception` that
    treats anything it catches as a task failure. That handler dumps its full log pools
    *unconditionally*, ignoring `suppress_log_files`: ~370 KB of "Failed to finish task"
    logs per abandoned episode, which over a training run is gigabytes of noise that also
    buries genuine failures. Inheriting from `BaseException` lets our teardown pass
    straight through that handler untouched.
    """


class SimulationFailed(RuntimeError):
    """RuFaS aborted the run.

    Raised for both routes a failure can take: an exception escaping the worker thread,
    and — far more insidiously — a task that RuFaS caught internally. `TaskManager` wraps
    every task in a broad `except Exception`, records the failure to the output manager's
    error pool, and returns *normally*. From the outside that is indistinguishable from a
    completed horizon, so without this check a crashed simulation would surface as a
    healthy `terminated=True` with reward 0. During RL training that fabricates terminal
    states the agent then learns from.
    """


#: Substrings of the error keys `TaskManager` writes when a task aborts.
_TASK_FAILURE_MARKERS = ("Failed to finish task", "Task(s) failed")


def _task_failures() -> list[str]:
    """Task-level failures RuFaS recorded during the run, if any."""
    from RUFAS.output_manager import OutputManager

    pool = getattr(OutputManager(), "errors_pool", None) or {}
    return [key for key in pool if any(m in key for m in _TASK_FAILURE_MARKERS)]


def _task_failure_causes() -> list[str]:
    """The underlying exception messages behind those failures.

    `TaskManager` records them as "Failed to recover from error: <msg>; traceback: ...".
    Without this the only visible text is "Task(s) failed", which is how the real crash
    cause (negative manure ammoniacal N) stayed hidden through a whole training run.
    """
    import re

    from RUFAS.output_manager import OutputManager

    pool = getattr(OutputManager(), "errors_pool", None) or {}
    causes = []
    for key, value in pool.items():
        if any(m in key for m in _TASK_FAILURE_MARKERS):
            causes += re.findall(r"Failed to recover from error: (.*?); traceback", str(value))
    return causes


class ThreadedPauseStepper:
    """Runs one RuFaS simulation, pausing it at each decision boundary.

    Usage is a strict alternation — the engine handed back is live and paused, so it may
    be read (and its levers mutated) only between `start`/`advance` calls:

        stepper = ThreadedPauseStepper(meta, work_dir, cadence="monthly")
        engine = stepper.start()          # paused at the first boundary
        while engine is not None:
            ...                           # read state, apply an action
            engine = stepper.advance()    # resume; None once the horizon ends
        pool = stepper.variables_pool()
        stepper.close()
    """

    def __init__(
        self,
        metadata_path: Path,
        work_dir: Path,
        cadence: str | BoundaryPredicate = "monthly",
        start_kwargs: dict | None = None,
    ) -> None:
        self._metadata_path = metadata_path
        self._work_dir = work_dir
        self._is_boundary = get_cadence(cadence) if isinstance(cadence, str) else cadence
        self._start_kwargs = start_kwargs
        self._to_driver: queue.Queue = queue.Queue()
        self._to_worker: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._originals: dict[str, Callable] = {}
        self._error: BaseException | None = None
        self._finished = False
        self._n_pauses = 0

    # -- properties ------------------------------------------------------------

    @property
    def finished(self) -> bool:
        """True once the simulation has run past its final boundary."""
        return self._finished

    @property
    def n_pauses(self) -> int:
        return self._n_pauses

    # -- lifecycle -------------------------------------------------------------

    def start(self):
        """Launch the simulation and run it to the first boundary.

        Returns the live, paused `SimulationEngine`, or None if the run finished without
        ever hitting a boundary (which means the cadence predicate never matched).
        """
        if self._thread is not None:
            raise RuntimeError("Stepper already started")

        from RUFAS.simulation_engine import SimulationEngine
        from RUFAS.task_manager import TaskManager

        from .runtime import start_kwargs as default_start_kwargs

        kwargs = self._start_kwargs or default_start_kwargs(self._work_dir)

        self._install_hooks(SimulationEngine)

        def worker() -> None:
            try:
                TaskManager().start(metadata_path=self._metadata_path, **kwargs)
            except _Teardown:
                pass  # expected: close() unwinding us
            except BaseException as exc:  # noqa: BLE001 — must reach the driver thread
                self._error = exc
            finally:
                self._to_driver.put(("done", None))

        self._thread = threading.Thread(target=worker, name="rufas-sim", daemon=True)
        self._thread.start()
        return self._wait_for_boundary()

    def advance(self):
        """Resume the simulation and run it to the next boundary.

        Any action must already have been applied — the worker is released immediately.
        Returns the paused engine, or None once the horizon ends.
        """
        if self._thread is None:
            raise RuntimeError("Stepper not started")
        if self._finished:
            return None
        self._to_worker.put("GO")
        return self._wait_for_boundary()

    def variables_pool(self) -> dict:
        """The run's flat `variables_pool`.

        Readable at any pause — it accumulates as the run proceeds — but note most soil
        variables are written by *annual* reporters, so mid-year they lag reality. Live
        engine state (see `observers.py`) is the truthful mid-run source.
        """
        from RUFAS.output_manager import OutputManager

        return OutputManager()._get_flat_variables_pool()

    def close(self) -> None:
        """Tear the worker down and restore RuFaS's methods. Safe to call twice."""
        try:
            if self._thread is not None and self._thread.is_alive() and not self._finished:
                self._to_worker.put("STOP")
                # Drain until the worker signals it has unwound.
                while True:
                    kind, _ = self._to_driver.get(timeout=60)
                    if kind == "done":
                        break
                    self._to_worker.put("STOP")
            if self._thread is not None:
                self._thread.join(timeout=60)
        finally:
            self._restore_hooks()
            self._thread = None

    def __enter__(self) -> "ThreadedPauseStepper":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internals -------------------------------------------------------------

    def _install_hooks(self, engine_cls) -> None:
        """Wrap every dispatchable daily-step method with the blocking hook.

        Patching the *class* before construction is fine: the engine's per-instance
        dispatch dict is built from bound methods at construction time, so it picks up
        the patched function.
        """
        for name in DAILY_METHODS:
            original = getattr(engine_cls, name, None)
            if original is None:
                continue  # tolerate RuFaS versions with a different set of methods
            self._originals[name] = original
            setattr(engine_cls, name, self._make_hook(original))

    def _make_hook(self, original: Callable) -> Callable:
        to_driver, to_worker, is_boundary = self._to_driver, self._to_worker, self._is_boundary

        def hooked(engine_self):
            if is_boundary(engine_self):
                to_driver.put(("boundary", engine_self))
                if to_worker.get() == "STOP":
                    raise _Teardown()
            return original(engine_self)

        return hooked

    def _restore_hooks(self) -> None:
        if not self._originals:
            return
        from RUFAS.simulation_engine import SimulationEngine

        for name, original in self._originals.items():
            setattr(SimulationEngine, name, original)
        self._originals.clear()

    def _wait_for_boundary(self):
        kind, payload = self._to_driver.get()
        if kind == "done":
            self._finished = True
            if self._thread is not None:
                self._thread.join(timeout=60)
            if self._error is not None:
                raise SimulationFailed("RuFaS raised inside the worker thread") from self._error
            failures = _task_failures()
            if failures:
                raise SimulationFailed(
                    "RuFaS aborted the run and caught the error internally, so the "
                    "simulation stopped early rather than completing its horizon. "
                    f"Recorded failures: {failures}. Cause: {_task_failure_causes()}"
                )
            return None
        self._n_pauses += 1
        return payload
