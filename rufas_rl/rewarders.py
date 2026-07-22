"""Scoring what happened between two pauses.

A rewarder is stateful: it snapshots the farm at each boundary and scores the *interval*
since the previous one, because reward in this MDP is a flow (milk produced, nitrogen
leached) rather than a level.

**On the default reward — read this before using a result.** `MilkMinusNitrogen` is a
bring-up proxy, not the paper's objective. It prices milk against nitrogen runoff with
flat coefficients and ignores feed cost, herd economics, and greenhouse gases entirely.
It exists so the vertical slice can be tested end-to-end and so an obviously-good action
can be shown to outrank an obviously-bad one. The real objective is RuFaS's EEE economics
plus enteric methane and field/manure emissions — those are reported annually, so a
faithful profit reward is sparse at monthly cadence and is deliberately left for later.
Swap it in through this same interface.
"""

from __future__ import annotations

from typing import Protocol

# Rough US placeholders, chosen only to put the two terms on a comparable scale.
DEFAULT_MILK_PRICE = 0.45  # $/kg milk
DEFAULT_N_RUNOFF_PENALTY = 5.0  # $/kg nitrate-N lost to runoff
#: Chosen so a monthly step on a ~100-cow herd scores O(1), which keeps PPO/SAC value
#: targets in a comfortable range. Purely a scaling convention, not economics.
REWARD_SCALE = 1e-5


class Rewarder(Protocol):
    def reset(self) -> None: ...

    def reward(self, engine) -> tuple[float, dict]: ...


def _annual_delta(current: float, previous: float) -> float:
    """Delta of a counter that RuFaS resets to zero each simulation year.

    A drop means the annual reset fired between the two pauses, so everything currently
    accumulated is new. This slightly under-counts the tail of the old year, which is
    acceptable for a proxy and disappears entirely under a yearly cadence.
    """
    return current if current < previous else current - previous


class MilkMinusNitrogen:
    """Milk revenue over the interval, less a penalty on nitrate runoff.

    Milk is integrated from the herd's instantaneous production rate, which RuFaS exposes
    only as a snapshot at each boundary. We use the trapezoid of the rates at both ends of
    the interval rather than the leading rate alone. That matters at the very first
    boundary, where `herd_statistics` has not been populated yet and the rate reads 0 —
    integrating from the leading edge would credit the opening month with no milk at all.
    The first interval is still somewhat under-counted; a yearly cadence, or a rewarder
    built on RuFaS's own reported milk totals, avoids the issue entirely.
    """

    def __init__(
        self,
        milk_price: float = DEFAULT_MILK_PRICE,
        n_runoff_penalty: float = DEFAULT_N_RUNOFF_PENALTY,
        scale: float = REWARD_SCALE,
    ) -> None:
        self.milk_price = milk_price
        self.n_runoff_penalty = n_runoff_penalty
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self._prev_day: float | None = None
        self._prev_milk_rate: float = 0.0
        self._prev_runoff_n: dict[int, float] = {}

    def _runoff_n(self, engine) -> dict[int, float]:
        out: dict[int, float] = {}
        field_manager = getattr(engine, "field_manager", None)
        for idx, field in enumerate(getattr(field_manager, "fields", []) or []):
            data = getattr(getattr(field, "soil", None), "data", None)
            value = getattr(data, "annual_runoff_nitrates_total", 0.0) or 0.0
            size = getattr(getattr(field, "field_data", None), "field_size", 1.0) or 1.0
            out[idx] = float(value) * float(size)  # per-ha -> field total
        return out

    def reward(self, engine) -> tuple[float, dict]:
        stats = getattr(getattr(engine, "herd_manager", None), "herd_statistics", None)
        milk_rate = float(getattr(stats, "daily_milk_production", 0.0) or 0.0)
        day = float(getattr(engine.time, "simulation_day", 0.0) or 0.0)
        runoff_n = self._runoff_n(engine)

        if self._prev_day is None:  # first boundary: nothing has elapsed yet
            days = 0.0
            milk_kg = 0.0
            n_lost = 0.0
        else:
            days = max(day - self._prev_day, 0.0)
            milk_kg = 0.5 * (self._prev_milk_rate + milk_rate) * days
            n_lost = sum(
                _annual_delta(runoff_n.get(k, 0.0), self._prev_runoff_n.get(k, 0.0))
                for k in runoff_n
            )

        self._prev_day = day
        self._prev_milk_rate = milk_rate
        self._prev_runoff_n = runoff_n

        revenue = milk_kg * self.milk_price
        penalty = n_lost * self.n_runoff_penalty
        reward = (revenue - penalty) * self.scale

        return reward, {
            "days": days,
            "milk_kg": milk_kg,
            "milk_revenue": revenue,
            "nitrate_runoff_kg": n_lost,
            "nitrate_penalty": penalty,
        }


class Profit:
    """Milk revenue less RuFaS's own feed cost, less an optional nitrate penalty.

    This is the reward that gives the ration lever economic meaning. Measured on the
    default scenario, reversing the lactating ration raises feed cost by **+122%** while
    milk moves **-0.2%** — a pure feed-substitution effect that `MilkMinusNitrogen` is
    completely blind to. So a policy under this reward is trading feed dollars against
    milk dollars, which is the real decision.

    Where each term comes from — and why:

    * **feed cost** is RuFaS's own dollar figure. `FeedManager.purchase_feed` prices each
      purchase against the scenario's cost data and reports it to the output pool as
      `...ration_interval_<feed>_cost`. We read those from the pool (they are not kept on
      a live object) and diff them across the interval. That means the *cost* side of the
      reward is entirely RuFaS's economics, not ours.
    * **milk revenue** is the one external number: total milk (kg) × an exogenous market
      milk price. RuFaS does not decide the milk price, so supplying it is legitimate.
      Crucially the *ranking* of rations is insensitive to it — a cheaper ration at equal
      milk wins at any positive price — so the headline result does not hinge on the
      price we pick.
    * **nitrate penalty** is optional and off by default here, so a first result is a
      clean profit number rather than a blended objective.

    Feed cost needs the output pool, which the stepper exposes; the rewarder pulls it from
    the `OutputManager` singleton it shares with the running simulation.
    """

    #: Pool keys of the form `...ration_interval_<feed>_cost`. `ration_interval` rather
    #: than `daily_feed_request` so purchases are counted once, on the buying cadence.
    _COST_MARKER = "ration_interval"

    def __init__(
        self,
        milk_price: float = DEFAULT_MILK_PRICE,
        n_runoff_penalty: float = 0.0,
        scale: float = REWARD_SCALE,
    ) -> None:
        self.milk_price = milk_price
        self.n_runoff_penalty = n_runoff_penalty
        self.scale = scale
        self._milk = MilkMinusNitrogen(
            milk_price=milk_price, n_runoff_penalty=n_runoff_penalty, scale=1.0
        )
        self.reset()

    def reset(self) -> None:
        self._milk.reset()
        self._prev_feed_cost: float | None = None

    def _feed_cost_to_date(self) -> float:
        """Total feed spend RuFaS has reported so far this run."""
        from RUFAS.output_manager import OutputManager

        pool = OutputManager()._get_flat_variables_pool()
        total = 0.0
        for key, value in pool.items():
            if self._COST_MARKER in key and key.endswith("_cost"):
                total += _sum_numbers(value)
        return total

    def reward(self, engine) -> tuple[float, dict]:
        # Milk revenue and N penalty over the interval (scale=1 so we combine in $).
        _, milk_info = self._milk.reward(engine)

        feed_cost_to_date = self._feed_cost_to_date()
        if self._prev_feed_cost is None:
            feed_cost = 0.0  # first boundary: charge nothing, just set the baseline
        else:
            feed_cost = max(feed_cost_to_date - self._prev_feed_cost, 0.0)
        self._prev_feed_cost = feed_cost_to_date

        revenue = milk_info["milk_revenue"]
        n_penalty = milk_info["nitrate_penalty"]
        profit = revenue - feed_cost - n_penalty

        return profit * self.scale, {
            **milk_info,
            "feed_cost": feed_cost,
            "profit": profit,
        }


def _sum_numbers(value) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, dict):
        return sum(_sum_numbers(v) for v in value.values())
    return 0.0


REWARDERS = {
    "milk_minus_nitrogen": MilkMinusNitrogen,
    "profit": Profit,
}


def make_rewarder(name: str, **kwargs) -> Rewarder:
    try:
        return REWARDERS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown rewarder {name!r}; expected one of {sorted(REWARDERS)}") from None
