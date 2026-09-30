"""End-to-end tests that run a real RuFaS simulation.

Marked `slow` — each one spawns a subprocess, imports RuFaS, and simulates. Run with:

    pytest -m slow          # just these
    pytest -m "not slow"    # everything else
"""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl import EnvConfig, RufasEnv

pytestmark = pytest.mark.slow


@pytest.fixture
def env():
    # `failure_penalty` is set because these tests sample the action space at random, and
    # a substantial minority of random rations are extreme enough to crash RuFaS's manure
    # chemistry. That is a real property of the environment, not flakiness — any training
    # run that explores will hit it, which is why the penalty path exists.
    e = RufasEnv(EnvConfig(cadence="monthly", max_steps=2, failure_penalty=-1.0))
    yield e
    e.close()


def test_gymnasium_conformance(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)


def test_episode_runs_and_observations_stay_in_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "date" in info

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    # Normally truncation at max_steps; a random ration that crashes RuFaS terminates
    # instead, which is a legitimate outcome under `failure_penalty`.
    assert truncated or info.get("simulation_failed"), "episode neither truncated nor failed"


def test_time_advances_one_period_per_step(env):
    _, info = env.reset(seed=0)
    first = info["simulation_day"]
    _, _, _, _, info = env.step(env.neutral_action())
    assert info["simulation_day"] > first
    assert 28 <= info["days"] <= 31, "a monthly cadence step should span one month"


def test_same_seed_reproduces_the_same_start(env):
    """Reset hygiene (V3): episodes must be isolated across subprocesses."""
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    assert np.allclose(first, second)


def test_action_changes_the_trajectory():
    """Action plumbing (V4): a different ration must produce a different outcome."""

    def rollout(action_fn, steps=3):
        e = RufasEnv(EnvConfig(cadence="monthly", max_steps=steps))
        try:
            e.reset(seed=7)
            rewards = []
            for _ in range(steps):
                _, r, term, trunc, _ = e.step(action_fn(e))
                rewards.append(r)
                if term or trunc:
                    break
            return rewards
        finally:
            e.close()

    even = rollout(lambda e: e.neutral_action())
    # A materially different but *feasible* ration. Deliberately not an extreme one:
    # concentrating the diet onto a single feed crashes RuFaS's manure chemistry, and a
    # test that relied on that would be asserting on a failure rather than on dynamics.
    other = _draw(3, 1)
    skewed = rollout(lambda e: other)
    assert not np.allclose(even, skewed), "ration changes had no effect on the outcome"


def _draw(seed: int, index: int, size: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(index + 1):
        vec = rng.uniform(-1, 1, size=size).astype(np.float32)
    return vec


#: An extreme-but-valid ration that crashes RuFaS's manure ammonia calculation *when the
#: protein floor is off*: its lactating diet is too low in crude protein, so urine N goes
#: negative. Regression anchor: this used to surface as a healthy terminal step with
#: reward 0 instead of as a failure. The failure-path tests switch the floor off
#: (`min_crude_protein={}`) so they still exercise a real crash.
_CRASHING_ACTION = _draw(0, 2)
_NO_FLOOR = {}


def test_simulation_failure_is_not_disguised_as_a_normal_episode_end():
    from rufas_rl.stepper import SimulationFailed

    e = RufasEnv(EnvConfig(max_steps=3, min_crude_protein=_NO_FLOOR))
    try:
        e.reset(seed=11)
        with pytest.raises(Exception) as excinfo:
            e.step(_CRASHING_ACTION)
        # Crosses a process boundary, so the type is wrapped; the cause must survive.
        message = str(excinfo.value)
        assert isinstance(excinfo.value, SimulationFailed) or "SimulationFailed" in message, (
            f"failure was not reported as such: {message[:200]}"
        )
    finally:
        e.close()


def test_failure_penalty_ends_the_episode_instead_of_raising():
    e = RufasEnv(EnvConfig(max_steps=3, failure_penalty=-10.0, min_crude_protein=_NO_FLOOR))
    try:
        e.reset(seed=11)
        _, reward, terminated, _, info = e.step(_CRASHING_ACTION)
        assert reward == -10.0
        assert terminated
        assert info["simulation_failed"] is True
    finally:
        e.close()


def test_close_is_idempotent(env):
    env.reset(seed=0)
    env.close()
    env.close()


def test_protein_floor_keeps_the_crashing_action_alive():
    """The same action that crashes above survives once the floor lifts its protein."""
    e = RufasEnv(EnvConfig(max_steps=3))
    try:
        e.reset(seed=11)
        for _ in range(3):
            _, _, terminated, truncated, info = e.step(_CRASHING_ACTION)
            assert not info.get("simulation_failed")
        assert truncated and not terminated
    finally:
        e.close()
