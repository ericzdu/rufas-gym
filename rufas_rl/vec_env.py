"""A non-daemonic SubprocVecEnv so RufasEnv can run in parallel.

SB3's `SubprocVecEnv` starts its worker processes with `daemon=True`, and a daemonic
process may not spawn children. But every `RufasEnv` episode spawns its own fresh
subprocess (the one-episode-per-process rule that keeps RuFaS's global state from
degrading ~5x across runs). Nesting those under the stock SubprocVecEnv raises
"daemonic processes are not allowed to have children".

This subclass reruns the same setup with `daemon=False`. The only cost of non-daemonic
workers is that they are not force-killed if the parent dies abnormally; `close()` shuts
them down cleanly in the normal path, and each episode subprocess is itself daemonic and
short-lived, so a leak window is small.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _worker


class NonDaemonSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose workers are non-daemonic (so they can spawn episode processes)."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # "spawn" (not fork/forkserver): the workers must not inherit an already-imported
        # RuFaS, matching the isolation the episode subprocesses rely on.
        if start_method is None:
            start_method = "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=False is the whole point: these workers spawn the per-episode
            # subprocess, which daemonic processes are forbidden to do.
            process = ctx.Process(target=_worker, args=args, daemon=False)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
