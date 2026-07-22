"""A dynamics-preserving Gymnasium environment wrapping the RuFaS dairy simulator.

    from rufas_rl import RufasEnv, EnvConfig

    env = RufasEnv(EnvConfig(cadence="monthly", max_steps=12))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

RuFaS itself is never modified; see `stepper.py` (the pause) and `appliers.py` (the
action) for how that is achieved and what has been proven.
"""

from .config import EnvConfig
from .spec import ScenarioSpec, load_spec

__all__ = ["EnvConfig", "RufasEnv", "ScenarioSpec", "load_spec"]

__version__ = "0.1.0"


def __getattr__(name: str):
    # `RufasEnv` pulls in gymnasium; keep the import lazy so `load_spec` and the episode
    # worker remain usable without it installed.
    if name == "RufasEnv":
        from .env import RufasEnv

        return RufasEnv

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
