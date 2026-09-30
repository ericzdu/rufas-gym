"""One episode, driven in-process.

This is the whole MDP in one object: it owns the stepper (the pause), the appliers (the
action), the observer (the state) and the rewarder (the score). `RufasEnv` normally runs
this inside a dedicated subprocess, but it is a complete env on its own and is much
easier to debug directly — no IPC in the way.

Timing, which decides what an action actually means:

    the hook fires BEFORE the day is simulated

so a pause dated 1 July hands over the farm with June complete and July not yet begun. An
action applied there governs July onward, and the observation is the state entering July.
That makes `step` a clean `(s_t, a_t) -> (s_t+1, r_t)`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from .appliers import PriceApplier, get_applier, set_engine
from .bootstrap import ensure_importable
from .composite import CompositeImplementer, FIELD_LEVERS
from .config import EnvConfig
from .observers import FarmObserver
from .prices import build_price_features, make_price_path
from .rewarders import make_rewarder
from .runtime import flush_singletons, prepare_metadata
from .spec import ScenarioSpec, load_spec
from .stepper import SimulationFailed, ThreadedPauseStepper


class EpisodeError(RuntimeError):
    pass


class Episode:
    """A single RuFaS simulation, stepped at decision boundaries."""

    def __init__(self, config: EnvConfig, spec: ScenarioSpec | None = None) -> None:
        ensure_importable()
        self.config = config
        self.spec = spec or load_spec(config.task_metadata_path, config.task_index)
        self.implementer = CompositeImplementer(
            self.spec, config.levers, config.min_crude_protein
        )
        self.observer = FarmObserver(
            n_fields=self.spec.n_fields,
            price_features=build_price_features(self.spec, config),
        )
        self.rewarder = make_rewarder(config.rewarder, **config.rewarder_kwargs)
        # One applier per lever. Field-lever appliers are stateful (cache a per-episode
        # baseline), so they are rebuilt on each start().
        self.appliers: dict = {}

        self._stepper: ThreadedPauseStepper | None = None
        self._work_dir: Path | None = None
        self._engine = None
        self._last_obs: np.ndarray | None = None
        self._steps = 0
        self._done = False
        # Exogenous prices. Built per-episode in start() so the path is a function of the
        # episode seed; the applier is stateless and can be shared.
        self.price_path = None
        self.price_applier = PriceApplier()

    # -- lifecycle -------------------------------------------------------------

    def start(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Run to the first decision boundary and return the initial observation."""
        self.close()  # a previous episode in this process must be fully torn down

        flush_singletons()
        self._work_dir = Path(tempfile.mkdtemp(prefix="rufas_episode_"))
        metadata = prepare_metadata(self.config.task_metadata_path, self._work_dir, seed=seed)

        self._stepper = ThreadedPauseStepper(
            metadata_path=metadata,
            work_dir=self._work_dir,
            cadence=self.config.cadence,
        )
        self.rewarder.reset()
        self.appliers = {lever: get_applier(lever) for lever in self.config.levers}
        # The price path is drawn from the episode seed, so a seed pins the market as
        # well as RuFaS's own RNG — two episodes with the same seed are identical runs.
        self.price_path = make_price_path(
            self.spec,
            process=self.config.price_process,
            n_months=self.spec.n_years * 12,
            seed=seed,
            levels=self.config.price_levels,
            **self.config.price_kwargs,
        )
        self._steps = 0
        self._done = False

        engine = self._stepper.start()
        if engine is None:
            raise EpisodeError(
                f"The simulation finished without ever reaching a {self.config.cadence!r} "
                "boundary. The cadence predicate never matched — check the scenario's "
                "date range."
            )
        self._engine = engine

        # Prime the rewarder so the first interval is measured from here, and discard
        # the reward it returns (nothing has elapsed yet).
        self.rewarder.reward(engine)
        self._last_obs = self._observe(engine)
        return self._last_obs, self._info(engine)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply an action at the current pause, resume, and stop at the next boundary."""
        if self._stepper is None:
            raise EpisodeError("Episode not started; call start() first")
        if self._done:
            raise EpisodeError("Episode already finished; call start() again")

        decoded = self.implementer.decode(action)
        # The engine paused here is the one appliers mutate; field appliers reach it via
        # the module-level registry rather than an argument.
        set_engine(self._engine)
        is_year_boundary = self._engine.time.current_date.month == 1

        # Prices for the interval we are about to simulate, applied *before* resuming so
        # RuFaS formulates and buys at them. The rewarder is moved to the same milk price
        # and left there until after `advance()`, so the reward scored at the next pause
        # values the interval at the prices that were actually in force during it.
        prices = self.price_path.at(self._month_index(self._engine))
        self.price_applier.apply(prices)
        set_milk_price = getattr(self.rewarder, "set_milk_price", None)
        if set_milk_price is not None:
            set_milk_price(prices.milk)

        for lever, applier in self.appliers.items():
            # Rations can be reset at every boundary; field levers only at year starts,
            # because their events are keyed by year and re-applying mid-year would just
            # rebuild the same future events.
            if lever in FIELD_LEVERS and not is_year_boundary:
                continue
            applier.apply(decoded[lever])

        try:
            engine = self._stepper.advance()
        except SimulationFailed as exc:
            if self.config.failure_penalty is None:
                raise
            # End the episode as a penalised failure rather than taking down training.
            self._steps += 1
            self._done = True
            info = self._info(self._engine) if self._engine is not None else {}
            info.update(simulation_failed=True, action_decoded=decoded,
                        failure_reason=str(exc.__cause__ or exc))
            obs = self._last_obs
            if obs is None:
                obs = np.zeros(self.observer.size, dtype=np.float32)
            return obs, float(self.config.failure_penalty), True, False, info

        self._steps += 1

        if engine is None:
            # The horizon ended. The engine object graph outlives the worker thread, so
            # the final state is still readable through our existing reference.
            engine = self._engine
            terminated = True
        else:
            self._engine = engine
            terminated = False

        reward, reward_info = self.rewarder.reward(engine)
        # Observed *after* advancing, so the price block shows the upcoming month's
        # prices — the ones this observation's action will be paid at — while `reward`
        # above was scored at the interval's own prices.
        obs = self._observe(engine)
        self._last_obs = obs

        truncated = False
        if not terminated and self.config.max_steps is not None:
            if self._steps >= self.config.max_steps:
                truncated = True

        self._done = terminated or truncated
        info = self._info(engine)
        info.update(reward_info)
        info["action_decoded"] = decoded
        # The prices this interval was scored at — not the next interval's. Diagnostics
        # and the price-response check both need to line up prices with the reward.
        info["prices"] = {"milk": prices.milk, "feeds": dict(prices.feeds)}
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Tear down the worker thread and clean up the episode's scratch directory."""
        if self._stepper is not None:
            try:
                self._stepper.close()
            finally:
                self._stepper = None
        if self._work_dir is not None and not self.config.keep_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
        self._engine = None
        self._done = True

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- introspection ---------------------------------------------------------

    @property
    def observation_size(self) -> int:
        return self.observer.size

    @property
    def action_size(self) -> int:
        return self.implementer.size

    def _observe(self, engine) -> np.ndarray:
        """Observation at the current pause, price block included when configured."""
        block = None
        if self.observer.price_features is not None and self.price_path is not None:
            block = self.observer.price_features.extract(
                self.price_path, self._month_index(engine)
            )
        return self.observer.observe(engine, price_block=block)

    def _month_index(self, engine) -> int:
        """Months elapsed since the scenario's start — the price path's index.

        Derived from the simulation calendar rather than from `self._steps` so it stays
        correct under any cadence. Note that a yearly cadence therefore holds one month's
        prices for a whole simulated year; prices are a monthly series, so the price lever
        is only fully meaningful at monthly cadence or finer.
        """
        date = engine.time.current_date
        return (date.year - self.spec.start_year) * 12 + (date.month - 1)

    def _info(self, engine) -> dict:
        return {
            "date": str(engine.time.current_date),
            "simulation_day": int(getattr(engine.time, "simulation_day", 0) or 0),
            "simulation_year": int(getattr(engine.time, "current_simulation_year", 0) or 0),
            "step": self._steps,
            "n_pauses": self._stepper.n_pauses if self._stepper else 0,
        }
