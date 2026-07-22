"""Episode configuration.

Kept to plain primitives so it can be pickled across the process boundary to the episode
worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    #: Task metadata, relative to the RuFaS root (or absolute).
    task_metadata_path: str = "input/task_manager_metadata.json"
    #: Which task in that file to run as the episode.
    task_index: int = 0
    #: Decision cadence: "monthly", "yearly", or "daily".
    cadence: str = "monthly"
    #: Truncate the episode after this many steps. None runs the scenario's full horizon.
    max_steps: int | None = None
    #: Levers the agent controls per step. Only "rations" has a proven applier.
    levers: tuple[str, ...] = ("rations",)
    #: Reward preset name from `rewarders.REWARDERS`.
    rewarder: str = "milk_minus_nitrogen"
    rewarder_kwargs: dict = field(default_factory=dict)
    #: Keep the episode's temp directory around after it ends (for debugging).
    keep_work_dir: bool = False
    #: What to do when RuFaS aborts mid-episode. Some extreme-but-valid rations crash
    #: RuFaS's manure chemistry, and during training an unlucky action should not kill
    #: the whole job. Set a (negative) reward to end the episode with that penalty and
    #: `info["simulation_failed"] = True`. Left as None, the failure raises — the right
    #: default outside training, where a silent crash must not pass as a real terminal.
    failure_penalty: float | None = None

    #: Field levers (fertilizer, manure) change only at year boundaries; rations change at
    #: every boundary. This mask is applied by the episode.
    SUPPORTED_LEVERS = ("rations", "fertilizer", "manure")

    def __post_init__(self) -> None:
        if not self.levers:
            raise ValueError("At least one lever is required")
        unsupported = set(self.levers) - set(self.SUPPORTED_LEVERS)
        if unsupported:
            raise ValueError(
                f"No proven applier for lever(s) {sorted(unsupported)}. Supported: "
                f"{list(self.SUPPORTED_LEVERS)}. (crop rotation is not yet wired.)"
            )
