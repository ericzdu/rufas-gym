"""Non-learned policies to measure a learned one against.

The important one is `GreedyPriceRation`, and it exists because of what
`scripts/spike_prices.py` found. RuFaS's own formulation does not respond to prices, milk
is nearly insensitive to ration composition, and purchased quantities follow the ration
almost linearly. So profit is close to

    milk_revenue - sum_i price_i(t) * qty_i(ration)

a linear program in the ration with time-varying coefficients. Its solution is "put weight
on whatever is cheapest this month" — which a *fixed* ration provably cannot track, so PPO
should beat CMA-ES here. But that argument cuts both ways: if the optimal policy is
essentially a price ranking, a five-line heuristic captures most of the available gain and
"PPO beat the fixed-ration optimizer" stops being an interesting claim.

So this is the honest bar. PPO earns its keep only by beating a *tuned* greedy policy, and
what it would be beating it with is the part greedy ignores: nutrition constraints binding,
milk responding at the margin, feed inventory, and carryover between periods.
"""

from __future__ import annotations

import numpy as np

from .implementers import ACTION_LIMIT, RationImplementer
from .prices import PriceVector
from .spec import ScenarioSpec


class GreedyPriceRation:
    """Each period, tilt every ration toward its cheapest feeds.

    Feeds in a ration are ranked by current price and mapped linearly onto the action box:
    the cheapest gets `+sharpness`, the dearest `-sharpness`, the rest spaced evenly
    between. The implementer's softmax and inclusion caps then turn that into a valid
    ration, so this policy cannot emit anything PPO could not also emit — the two are
    compared over exactly the same action set.

    `sharpness` in [0, 1] is the one knob, and it is a real trade-off rather than a
    formality: at 0 the policy is an even split, at 1 it piles onto the single cheapest
    feed and will wreck the diet's nutrition. Tuning it (`tune_sharpness`) is what makes
    this a fair opponent instead of a straw man.

    Deliberately ranks on price alone, ignoring nutrition entirely. That is the point: it
    marks how much of the gain is available from naive price-following, so whatever PPO
    adds on top is attributable to everything greedy does not model.
    """

    def __init__(
        self,
        spec: ScenarioSpec,
        sharpness: float = 1.0,
        implementer: RationImplementer | None = None,
    ) -> None:
        if not 0.0 <= sharpness <= 1.0:
            raise ValueError(f"sharpness must be in [0, 1], got {sharpness}")
        self.spec = spec
        self.sharpness = float(sharpness)
        self.implementer = implementer or RationImplementer(spec)
        if not spec.ration_feed_ids:
            raise ValueError(
                "GreedyPriceRation needs per-ration feed IDs; this spec has none."
            )

    def action(self, prices: PriceVector) -> np.ndarray:
        """The action to play given this month's prices."""
        parts = []
        for feed_ids in self.spec.ration_feed_ids:
            costs = np.array(
                [prices.feeds.get(fid, np.inf) for fid in feed_ids], dtype=np.float64
            )
            parts.append(self._rank_to_action(costs))
        return np.concatenate(parts).astype(np.float32)

    def _rank_to_action(self, costs: np.ndarray) -> np.ndarray:
        n = len(costs)
        if n == 1:
            return np.zeros(1)
        # Rank rather than price magnitude, so one absurdly dear ingredient cannot swamp
        # the ordering among the rest — and so behaviour does not depend on price units.
        order = np.argsort(costs)
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(n)
        # rank 0 (cheapest) -> +1, rank n-1 (dearest) -> -1
        return (1.0 - 2.0 * rank / (n - 1)) * self.sharpness * ACTION_LIMIT

    def __repr__(self) -> str:
        return f"GreedyPriceRation(sharpness={self.sharpness:.2f})"


def tune_sharpness(evaluate, sharpnesses=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Pick the best `sharpness` under `evaluate(sharpness) -> profit`.

    Kept caller-driven because evaluating one setting means running whole episodes, and
    only the caller knows the horizon, seeds and price paths that make the comparison
    fair. Returns the winner plus the full sweep, since the shape of the curve is itself
    informative: a flat one says price-following buys nothing here.
    """
    results = {s: float(evaluate(s)) for s in sharpnesses}
    best = max(results, key=results.get)
    return {"best_sharpness": best, "best_profit": results[best], "sweep": results}
