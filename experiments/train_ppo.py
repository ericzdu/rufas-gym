#!/usr/bin/env python3
"""Train a PPO policy on the multi-lever RufasEnv.

This is the actual reinforcement-learning result: unlike the CMA-ES experiment (which
optimizes one fixed action), PPO learns a *policy* that reads the farm's state each period
and chooses that period's rations, fertilizer and manure — so it can, in principle, react
to soil-N carryover across years.

    # short smoke run — proves the loop learns, ~1-2h
    python experiments/train_ppo.py --timesteps 30000 --years 7 --n-envs 4

    # full run — overnight
    python experiments/train_ppo.py --timesteps 300000 --years 7 --n-envs 8

Each env steps a full RuFaS simulation, so timesteps are expensive: at monthly cadence a
7-year episode yields 84 transitions, so 30k steps is ~360 episodes. Parallel envs
(`--n-envs`) are the main throughput lever; each is its own subprocess.

Baselines for comparison (from the CMA-ES experiment) are the farm's Midwest-default
ration and the CMA-ES optimum; this script logs the learned policy's return so it can be
placed against them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def make_env_fn(years: int, seed: int):
    from stable_baselines3.common.monitor import Monitor

    from rufas_rl import EnvConfig, RufasEnv

    def _make():
        env = RufasEnv(EnvConfig(
            cadence="monthly",
            max_steps=years * 12,
            rewarder="profit",
            levers=("rations", "fertilizer", "manure"),
            failure_penalty=-50.0,  # scaled reward units; a crashed episode is a big loss
        ))
        env.reset(seed=seed)
        return Monitor(env)  # records episode returns for the learning curve

    return _make


def _episode_return_callback(store: list):
    """SB3 callback that records each finished episode's return and profit."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Recorder(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:  # Monitor writes this at episode end
                    store.append({"t": self.num_timesteps, "return": float(ep["r"])})
            return True

    return _Recorder()


def _eval_env(years: int, seed: int, stats_path: Path):
    """A single normalized eval env loaded with the training VecNormalize stats.

    Observations are normalized with the trained statistics (so the policy sees what it
    trained on); reward normalization is off and stats are frozen, so `info['profit']`
    stays in real dollars.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env_fn(years, seed)])
    venv = VecNormalize.load(str(stats_path), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _rollout_profit(venv, act_fn, episodes: int) -> dict:
    """Run whole episodes on a vec env; return mean raw profit and (normalized) return."""
    returns, profits = [], []
    for _ in range(episodes):
        obs = venv.reset()
        ret = profit = 0.0
        done = False
        while not done:
            action = act_fn(obs)
            obs, reward, dones, infos = venv.step(action)  # SB3 vec API: 4-tuple
            ret += float(reward[0])
            profit += float(infos[0].get("profit", 0.0))
            done = bool(dones[0])  # vec env auto-resets on done
        returns.append(ret)
        profits.append(profit)
    return {"mean_return": float(np.mean(returns)), "mean_profit": float(np.mean(profits))}


def evaluate(model, years: int, seed: int, stats_path: Path, episodes: int = 3) -> dict:
    """Mean profit of the learned policy under deterministic actions."""
    venv = _eval_env(years, seed, stats_path)
    try:
        return _rollout_profit(venv, lambda obs: model.predict(obs, deterministic=True)[0], episodes)
    finally:
        venv.close()


def evaluate_neutral(years: int, seed: int, stats_path: Path, episodes: int = 3) -> dict:
    """Baseline: the neutral action (even ration, baseline-rate fertilizer/manure)."""
    venv = _eval_env(years, seed, stats_path)
    try:
        neutral = venv.venv.envs[0].unwrapped.neutral_action()
        return _rollout_profit(venv, lambda obs: np.array([neutral]), episodes)
    finally:
        venv.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--years", type=int, default=7)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ppo")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rufas_rl.vec_env import NonDaemonSubprocVecEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env_fn(args.years, args.seed + i) for i in range(args.n_envs)]
    if args.n_envs == 1:
        vec = DummyVecEnv(env_fns)  # no reason to pay IPC for a single env
    else:
        # Our NonDaemonSubprocVecEnv (not SB3's SubprocVecEnv) because each RufasEnv
        # episode spawns its own subprocess, which SB3's daemonic workers may not do.
        vec = NonDaemonSubprocVecEnv(env_fns)

    # Normalize observations and returns. The 4k-step run left the value function unable
    # to fit (explained_variance ~0) — observations span ±10 and returns ~5, and PPO's
    # value head learns far better on standardized targets. clip keeps outliers bounded.
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # Rollout spans ~4 episodes/env (each episode is years*12 monthly steps); batch_size
    # divides the rollout so PPO's minibatching is clean.
    n_steps = args.years * 12 * 4
    model = PPO(
        "MlpPolicy", vec, seed=args.seed, verbose=1,
        n_steps=n_steps, batch_size=n_steps // 3, gae_lambda=0.95, gamma=0.99,
        ent_coef=0.01, tensorboard_log=str(out / "tb"),
    )

    print(f"Training PPO: {args.timesteps} steps, {args.n_envs} envs, {args.years}-yr episodes")
    curve: list = []
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False,
                callback=_episode_return_callback(curve))
    train_min = (time.time() - t0) / 60

    model.save(out / "ppo_rufas")
    stats_path = out / "vecnormalize.pkl"
    vec.save(str(stats_path))  # normalization stats needed to evaluate the policy
    vec.close()
    print(f"\nTrained in {train_min:.1f} min ({len(curve)} episodes). Evaluating...")

    learned = evaluate(model, args.years, seed=1000, stats_path=stats_path)
    baseline = evaluate_neutral(args.years, seed=1000, stats_path=stats_path)
    print(f"  learned policy : mean profit ${learned['mean_profit']:,.0f} "
          f"(return {learned['mean_return']:.2f})")
    print(f"  neutral policy : mean profit ${baseline['mean_profit']:,.0f} "
          f"(return {baseline['mean_return']:.2f})")
    gain = learned["mean_profit"] - baseline["mean_profit"]
    print(f"  learned - neutral: ${gain:+,.0f}")

    import json
    (out / "result.json").write_text(json.dumps({
        "train_min": train_min, "timesteps": args.timesteps, "n_envs": args.n_envs,
        "years": args.years, "n_episodes": len(curve),
        "learned": learned, "neutral": baseline, "gain_vs_neutral": gain,
        "curve": curve,
    }, indent=2))
    print(f"Saved model + result to {out}/")


if __name__ == "__main__":
    main()
