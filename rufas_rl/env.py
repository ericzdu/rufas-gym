"""`RufasEnv` — RuFaS as a Gymnasium environment.

Each episode is one multi-year RuFaS simulation, paused at every decision boundary. The
agent sees the farm's real carryover state at each pause and sets the next period's
levers; the simulator runs its daily loop autonomously in between. The RL timestep is a
management decision, not RuFaS's daily tick.

    env = RufasEnv(EnvConfig(cadence="monthly"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Nothing in RuFaS is modified. The pause is a monkeypatched daily-step method that only
blocks; the action is applied by re-invoking RuFaS's own setup functions on live objects,
a path proven physically identical to configuring the scenario that way from the start.

**Levers.** Rations (per-boundary), fertilizer and manure (per-field N/P multipliers,
applied at year boundaries) are wired — each with a passing equivalence spike. Field
levers drive the cross-year soil-N/P carryover the project is about. Crop rotation is a
categorical lever and is not yet wired.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .composite import CompositeImplementer
from .config import EnvConfig
from .harness import EpisodeProcess
from .observers import OBS_CLIP, FarmObserver
from .prices import build_price_features
from .spec import ScenarioSpec, load_spec


class RufasEnv(gym.Env):
    """A sequential whole-farm management environment backed by RuFaS."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        spec: ScenarioSpec | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        # The spec comes from reading the scenario's JSON directly — no simulation — so
        # the spaces can exist before any episode has ever run.
        # NB: named `scenario_spec`, not `spec` — Gymnasium reserves `env.spec` for its
        # own `EnvSpec` and its checker trips over anything else living there.
        self.scenario_spec = spec or load_spec(
            self.config.task_metadata_path, self.config.task_index
        )
        self.render_mode = render_mode

        self._implementer = CompositeImplementer(
            self.scenario_spec, self.config.levers, self.config.min_crude_protein
        )
        self._observer = FarmObserver(
            n_fields=self.scenario_spec.n_fields,
            price_features=build_price_features(self.scenario_spec, self.config),
        )

        self.action_space = self._implementer.action_space()
        self.observation_space = spaces.Box(
            low=-OBS_CLIP,
            high=OBS_CLIP,
            shape=(self._observer.size,),
            dtype=np.float32,
        )

        self._process: EpisodeProcess | None = None
        self._steps = 0
        self._episode_return = 0.0

    # -- Gymnasium API ---------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # A fresh interpreter per episode: RuFaS's singletons and its ~5x in-process
        # slowdown make reuse both unsafe and pointless.
        self.close()
        self._process = EpisodeProcess(self.config, self.scenario_spec)
        obs, info = self._process.start(seed=seed)
        self._steps = 0
        self._episode_return = 0.0
        return np.asarray(obs, dtype=np.float32), dict(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._process is None:
            raise RuntimeError("Call reset() before step()")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self._process.step(action)

        self._steps += 1
        self._episode_return += float(reward)
        info = dict(info)
        info["episode_return"] = self._episode_return

        if terminated or truncated:
            # The horizon is over; free the interpreter rather than leave it parked.
            self.close()

        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None

    # -- convenience -----------------------------------------------------------

    @property
    def max_episode_steps(self) -> int:
        """Decision points in a full horizon, before any `max_steps` truncation."""
        return self.scenario_spec.n_boundaries(self.config.cadence)

    def neutral_action(self) -> np.ndarray:
        """An even split within each ration — a valid action, not a no-op."""
        return self._implementer.neutral_action()

    def observation_names(self) -> list[str]:
        """Per-element labels for the observation vector."""
        return self._observer.layout.names()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 — interpreter teardown must not raise
            pass
