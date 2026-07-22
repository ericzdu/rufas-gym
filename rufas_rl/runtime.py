"""Preparing a RuFaS run so it can be paused.

Two things have to be true before the pause hook works at all, and both are silent
failures if you get them wrong:

* **`parallel_workers` must be 1.** Above 1, RuFaS runs tasks through a
  `multiprocessing.Pool`, so the simulation happens in a *different process* and our
  in-process monkeypatch never applies. The run completes normally and simply never
  pauses.
* **`multi_run_counts` must be 1.** A multi-run task would execute the horizon several
  times over, and an episode is one horizon.

Rather than edit the user's scenario files, we copy the task JSON into the episode's
work directory with those fields forced, and point a copied metadata file at it. The
scenario on disk is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bootstrap import resolve


def prepare_metadata(
    task_metadata_path: str | Path,
    work_dir: Path,
    seed: int | None = None,
) -> Path:
    """Write a pausable copy of the task metadata into `work_dir`; return its path.

    Forces single-process, single-run execution. If `seed` is given it overrides every
    task's `random_seed`, which is how `env.reset(seed=...)` pins an episode.
    """
    meta_path = resolve(task_metadata_path)
    meta = json.loads(meta_path.read_text())

    tasks_path = resolve(meta["files"]["tasks"]["path"])
    tasks = json.loads(tasks_path.read_text())

    tasks["parallel_workers"] = 1
    for task in tasks.get("tasks", []):
        if task.get("task_type") == "SIMULATION_MULTI_RUN":
            task["multi_run_counts"] = 1
        if seed is not None:
            task["random_seed"] = int(seed)
        # Per-task settings win over the corresponding `TaskManager.start` kwargs, so
        # suppression has to be set *here* to take effect. Without it every episode
        # writes a ~470 KB metadata-properties dump — fine for one run, gigabytes across
        # a training job.
        task["suppress_log_files"] = True
        task["log_verbosity"] = "none"

    patched_tasks = work_dir / "tasks.json"
    patched_tasks.write_text(json.dumps(tasks, indent=2))

    meta["files"]["tasks"] = {**meta["files"]["tasks"], "path": str(patched_tasks.resolve())}
    patched_meta = work_dir / "task_manager_metadata.json"
    patched_meta.write_text(json.dumps(meta, indent=2))
    return patched_meta


def start_kwargs(work_dir: Path) -> dict:
    """`TaskManager.start` arguments for a quiet, self-contained episode run."""
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


def flush_singletons() -> None:
    """Clear RuFaS's InputManager/OutputManager singleton pools.

    Hygiene only. It does *not* make in-process reuse safe — a second run in the same
    interpreter is ~5x slower even after flushing, which is why episodes get their own
    process.
    """
    from RUFAS.input_manager import InputManager
    from RUFAS.output_manager import OutputManager

    InputManager().flush_pool()
    OutputManager().flush_pools()
