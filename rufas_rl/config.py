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
    #: Exogenous price process: "static" (the scenario's constants), "synthetic"
    #: (correlated mean-reverting paths, a fresh one per episode seed) or "real" (a
    #: resolved CSV). Measured effect: prices move the reward but *not* what RuFaS
    #: physically feeds — its least-cost formulation sits at a price-independent corner —
    #: so the agent's ration is the only price-responsive channel. See `prices.py`.
    #: Defaults to "static", which reproduces the environment as it behaved before the
    #: price lever existed.
    price_process: str = "static"
    #: Long-run price levels: "configured" (RuFaS's own placeholders) or "realistic"
    #: (plausible US dairy prices). The scenario's placeholders price every forage at
    #: $0.01/kg DM against $0.50 corn grain, which makes the profit optimum an artifact
    #: of the input file; anything meant as a result about dairy farming wants
    #: "realistic". See `prices.REALISTIC_PRICES`.
    price_levels: str = "configured"
    #: Extra arguments for the price path: `volatility_scale` for synthetic,
    #: `csv_path` for real.
    price_kwargs: dict = field(default_factory=dict)
    #: Whether the observation carries the price block. `None` means auto — observe prices
    #: exactly when they vary. Leave it on auto unless you are deliberately measuring a
    #: price-blind policy: varying prices the agent cannot see make the reward depend on
    #: hidden state, which breaks the Markov property and looks like a training failure.
    observe_prices: bool | None = None
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

    #: Price processes `price_process` accepts. "prices" is deliberately not a lever:
    #: it is the market, not a decision, and must not appear in `levers`.
    PRICE_PROCESSES = ("static", "synthetic", "real")
    PRICE_LEVELS = ("configured", "realistic")

    def __post_init__(self) -> None:
        if not self.levers:
            raise ValueError("At least one lever is required")
        if self.price_levels not in self.PRICE_LEVELS:
            raise ValueError(
                f"Unknown price_levels {self.price_levels!r}. Expected one of "
                f"{list(self.PRICE_LEVELS)}."
            )
        if self.price_process not in self.PRICE_PROCESSES:
            raise ValueError(
                f"Unknown price_process {self.price_process!r}. Expected one of "
                f"{list(self.PRICE_PROCESSES)}."
            )
        unsupported = set(self.levers) - set(self.SUPPORTED_LEVERS)
        if unsupported:
            raise ValueError(
                f"No proven applier for lever(s) {sorted(unsupported)}. Supported: "
                f"{list(self.SUPPORTED_LEVERS)}. (crop rotation is not yet wired.)"
            )
