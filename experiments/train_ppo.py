#!/usr/bin/env python3
"""Train a PPO policy on the multi-lever RufasEnv.

Unlike the CMA-ES experiment (which optimizes one *fixed* ration), PPO learns a policy that
reads the farm's state each period — including the current milk and feed prices — and sets
that period's ration. That difference is the whole experiment: under a price process no
fixed ration can track, a policy that sees prices should strictly dominate the best fixed
one, and the margin is the result.

Read `rufas_rl/prices.py` before interpreting any number out of here. In particular the
honest bar is not CMA-ES but `baselines.GreedyPriceRation`: profit is close to a linear
program in the ration with time-varying coefficients, so naive price-following already
captures much of the gain.

    # Stage 1 validation — does it learn, and does it follow prices? ~1h
    python experiments/train_ppo.py --timesteps 3000 --years 2 --n-envs 4

    # Stage 2 — multi-day
    python experiments/train_ppo.py --timesteps 200000 --years 7 --n-envs 8

Each env steps a full RuFaS simulation, so timesteps are expensive: at monthly cadence a
7-year episode yields 84 transitions, so 30k steps is ~360 episodes. Parallel envs
(`--n-envs`) are the main throughput lever; each is its own subprocess.

The run also reports `price_response`: the correlation between each feed's share of the
ration and its price. That is the acceptance test — a strong return from a policy that
ignores prices means the reformulation failed, however good the number looks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def make_env_config(years: int, price_process: str = "synthetic",
                    price_levels: str = "realistic", **kwargs):
    """The environment both PPO and every baseline must run under.

    One function so a baseline cannot accidentally be measured on a different problem —
    which is most of what went wrong the first time round. The defaults encode the
    fair-fight fixes:

    * `levers=("rations",)` — the CMA-ES baseline optimizes rations only. The previous run
      gave PPO fertilizer and manure as well: four extra dimensions that apply only at
      year boundaries and barely touch a purchased-feed-cost reward, so they were mostly
      exploration noise loaded onto one side of the comparison.
    * `price_process="synthetic"` — a fresh correlated price path per episode, so the
      policy sees unlimited price histories instead of memorizing one.
    * `price_levels="realistic"` — the scenario's own prices are placeholders that make
      "feed only corn silage" optimal; see `prices.REALISTIC_PRICES`.
    """
    from rufas_rl import EnvConfig

    return EnvConfig(
        cadence="monthly",
        max_steps=years * 12,
        rewarder="profit",
        levers=("rations",),
        price_process=price_process,
        price_levels=price_levels,
        failure_penalty=-50.0,  # scaled reward units; a crashed episode is a big loss
        **kwargs,
    )


def make_env_fn(years: int, seed: int, **config_kwargs):
    from stable_baselines3.common.monitor import Monitor

    from rufas_rl import RufasEnv

    def _make():
        env = RufasEnv(make_env_config(years, **config_kwargs))
        env.reset(seed=seed)
        return Monitor(env)  # records episode returns for the learning curve

    return _make


def warm_start(model, spec) -> None:
    """Start the policy at the farm's configured ration rather than at an even split.

    CMA-ES was seeded there and PPO was not, which is worth ~$337k of head start on the
    2-year horizon ($830k configured vs $494k even-split). Shifting the action network's
    output bias moves the policy's *mean* action to the configured ration while leaving
    the learned weights at their initialization, so exploration still starts wide and
    nothing about the optimizer changes — it simply starts from current practice, which is
    also the honest question ("can we beat what the farm does today?").
    """
    import torch

    configured = configured_action(spec)
    with torch.no_grad():
        bias = model.policy.action_net.bias
        if bias.shape[0] != configured.size:
            raise ValueError(
                f"Cannot warm-start: policy emits {bias.shape[0]} actions but the "
                f"configured ration encodes to {configured.size}. Are the levers "
                "rations-only?"
            )
        bias.copy_(torch.as_tensor(configured, dtype=bias.dtype))
        # Zero the last layer's weights so the initial mean action *is* the bias,
        # regardless of what the (random) feature extractor emits on the first states.
        model.policy.action_net.weight.mul_(0.0)


def configured_action(spec) -> np.ndarray:
    """The scenario's actual ration as an action vector (same helper as CMA-ES uses)."""
    import json

    from rufas_rl.bootstrap import resolve
    from rufas_rl.implementers import RationImplementer

    scenario_meta = json.loads(spec.scenario_metadata_path.read_text())
    feed = json.loads(resolve(scenario_meta["files"]["feed"]["path"]).read_text())
    pcts = [[f["ration_percentage"] for f in r["feeds"]] for r in feed["rations"]]
    return RationImplementer(spec).encode(pcts)


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


def _eval_env(years: int, seed: int, stats_path: Path, **config_kwargs):
    """A single normalized eval env loaded with the training VecNormalize stats.

    Observations are normalized with the trained statistics (so the policy sees what it
    trained on); reward normalization is off and stats are frozen, so `info['profit']`
    stays in real dollars.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env_fn(years, seed, **config_kwargs)])
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


def evaluate(model, years: int, seed: int, stats_path: Path, episodes: int = 3,
             **config_kwargs) -> dict:
    """Mean profit of the learned policy under deterministic actions."""
    venv = _eval_env(years, seed, stats_path, **config_kwargs)
    try:
        return _rollout_profit(venv, lambda obs: model.predict(obs, deterministic=True)[0], episodes)
    finally:
        venv.close()


def price_response(model, years: int, seed: int, stats_path: Path, spec,
                   episodes: int = 2, **config_kwargs) -> dict:
    """Does the policy's ration actually move with prices?

    **The Stage 1 acceptance test.** Return alone cannot answer it: a policy that ignored
    prices entirely and simply found a good fixed ration would still score well, and would
    still lose to CMA-ES for exactly the reasons the first run did. What has to be true for
    the reformulation to have worked is that the *same* policy feeds differently in a
    cheap-forage month than in a dear-forage month.

    For each feed, the Spearman correlation across months between its share of the ration
    and its price. Price-following means a negative correlation — dearer feed, smaller
    share. Spearman rather than Pearson because only the ordering is meaningful; the
    softmax makes the share-price relationship monotone but not linear.

    A mean correlation near zero means the policy is price-blind and no amount of further
    training will fix it — the problem, the observation or the reward is wrong.
    """
    from scipy.stats import spearmanr

    venv = _eval_env(years, seed, stats_path, **config_kwargs)
    shares: dict[int, list[float]] = {f: [] for f in spec.feed_ids}
    prices: dict[int, list[float]] = {f: [] for f in spec.feed_ids}
    try:
        for _ in range(episodes):
            obs = venv.reset()
            done = False
            while not done:
                action = model.predict(obs, deterministic=True)[0]
                obs, _, dones, infos = venv.step(action)
                info = infos[0]
                decoded = info.get("action_decoded")
                priced = info.get("prices")
                if decoded and priced:
                    # Average a feed's share across the rations it appears in, so one
                    # number per feed per month.
                    for fid in spec.feed_ids:
                        pcts = [
                            ration[ids.index(fid)]
                            for ration, ids in zip(decoded["rations"], spec.ration_feed_ids)
                            if fid in ids
                        ]
                        if pcts:
                            shares[fid].append(float(np.mean(pcts)))
                            prices[fid].append(float(priced["feeds"][fid]))
                done = bool(dones[0])
    finally:
        venv.close()

    per_feed = {}
    for fid in spec.feed_ids:
        x, y = prices[fid], shares[fid]
        # A feed whose price never moved, or whose share never moved, has no correlation
        # to report — np.std guards against spearmanr returning nan.
        if len(x) > 3 and np.std(x) > 0 and np.std(y) > 0:
            per_feed[fid] = float(spearmanr(x, y).statistic)
    mean_corr = float(np.mean(list(per_feed.values()))) if per_feed else float("nan")
    return {
        "per_feed_spearman": per_feed,
        "mean_spearman": mean_corr,
        "price_following": bool(mean_corr < -0.2),
    }


def evaluate_neutral(years: int, seed: int, stats_path: Path, episodes: int = 3,
                     **config_kwargs) -> dict:
    """Baseline: the neutral action (even ration, baseline-rate fertilizer/manure)."""
    venv = _eval_env(years, seed, stats_path, **config_kwargs)
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
    ap.add_argument("--no-warm-start", action="store_true",
                    help="start from a random policy instead of the configured ration")
    ap.add_argument("--price-process", default="synthetic",
                    choices=["static", "synthetic", "real"])
    ap.add_argument("--price-levels", default="realistic",
                    choices=["configured", "realistic"])
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rufas_rl import load_spec
    from rufas_rl.vec_env import NonDaemonSubprocVecEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"price_process": args.price_process, "price_levels": args.price_levels}
    env_fns = [make_env_fn(args.years, args.seed + i, **env_kwargs)
               for i in range(args.n_envs)]
    if args.n_envs == 1:
        vec = DummyVecEnv(env_fns)  # no reason to pay IPC for a single env
    else:
        # Our NonDaemonSubprocVecEnv (not SB3's SubprocVecEnv) because each RufasEnv
        # episode spawns its own subprocess, which SB3's daemonic workers may not do.
        vec = NonDaemonSubprocVecEnv(env_fns)

    # Normalize observations and returns. The 4k-step run left the value function unable
    # to fit (explained_variance ~0) — observations span ±10 and returns ~5, and PPO's
    # value head learns far better on standardized targets. clip keeps outliers bounded.
    # gamma=1.0 throughout: the objective is *total* profit over the horizon, and every
    # episode is a fixed, finite 12*years steps. Discounting at 0.99 valued month 24 at
    # 0.79 of month 1 — the agent was being asked to solve a different problem from the
    # one the CMA-ES baseline optimizes, and then compared against it.
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=1.0)

    # Rollout spans ~4 episodes/env (each episode is years*12 monthly steps); batch_size
    # divides the rollout so PPO's minibatching is clean.
    n_steps = args.years * 12 * 4
    # Stability settings. The first tuned run climbed to a good policy (return ~5, above
    # neutral) then *collapsed* back down — the classic PPO failure where an over-large
    # update wrecks the policy. The fixes, all standard: a smaller learning rate, a
    # `target_kl` that aborts an update once the policy has moved too far, and a small
    # entropy bonus so exploration does not keep pushing the policy off a good solution.
    model = PPO(
        "MlpPolicy", vec, seed=args.seed, verbose=1,
        n_steps=n_steps, batch_size=n_steps // 3, gae_lambda=0.95, gamma=1.0,
        learning_rate=1e-4, ent_coef=0.001, target_kl=0.03,
        tensorboard_log=str(out / "tb"),
    )
    if not args.no_warm_start:
        warm_start(model, load_spec())
        print("Policy warm-started at the farm's configured ration.")

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

    learned = evaluate(model, args.years, seed=1000, stats_path=stats_path, **env_kwargs)
    baseline = evaluate_neutral(args.years, seed=1000, stats_path=stats_path, **env_kwargs)
    print(f"  learned policy : mean profit ${learned['mean_profit']:,.0f} "
          f"(return {learned['mean_return']:.2f})")
    print(f"  neutral policy : mean profit ${baseline['mean_profit']:,.0f} "
          f"(return {baseline['mean_return']:.2f})")
    gain = learned["mean_profit"] - baseline["mean_profit"]
    print(f"  learned - neutral: ${gain:+,.0f}")

    # The acceptance test — a good return with a price-blind policy is a failed
    # reformulation, not a success.
    response = {"mean_spearman": float("nan"), "price_following": False}
    if args.price_process != "static":
        spec = load_spec()
        response = price_response(model, args.years, seed=1000, stats_path=stats_path,
                                  spec=spec, **env_kwargs)
        verdict = ("PRICE-FOLLOWING" if response["price_following"]
                   else "PRICE-BLIND — the policy is ignoring prices")
        print(f"  price response : mean Spearman(share, price) = "
              f"{response['mean_spearman']:+.3f}  -> {verdict}")

    import json
    (out / "result.json").write_text(json.dumps({
        "train_min": train_min, "timesteps": args.timesteps, "n_envs": args.n_envs,
        "years": args.years, "n_episodes": len(curve),
        "learned": learned, "neutral": baseline, "gain_vs_neutral": gain,
        "price_process": args.price_process, "price_levels": args.price_levels,
        "warm_started": not args.no_warm_start, "gamma": 1.0,
        "levers": ["rations"], "price_response": response,
        "curve": curve,
    }, indent=2))
    print(f"Saved model + result to {out}/")


if __name__ == "__main__":
    main()
