"""Exogenous input and output prices — the reason feedback is worth having.

Without this module the environment is a *static* optimization problem wearing an MDP's
clothes. Feed prices are constants, milk price is a constant, and milk output is nearly
insensitive to ration composition, so the profit-maximizing ration is a single fixed
corner of the simplex that never changes. A policy that reads state has nothing to read;
CMA-ES over one fixed action searches that corner directly and wins on every axis. That
is exactly what the overnight run measured.

Time-varying prices break the tie *structurally*. When the relative cost of forage and
concentrate moves month to month, the profit-maximizing ration moves with it, so the best
achievable fixed ration is only a compromise across price states, and a policy that
observes prices strictly dominates it. The margin between them is the experiment.

**The agent's ration is the only price-responsive channel — measured, not assumed.**
RuFaS's ration formulation looks like it should respond to prices on its own: every
`formulation_interval` days each pen solves `min sum(amounts * prices)`
(`ration_optimizer.py`) within `user_defined_ration_tolerance` of the ration we set. It
does not. `scripts/spike_prices.py` swung relative prices 30x (forage up 10x, corn grain
down 3x) and the kilograms RuFaS purchased were *bit-identical*; only the dollars moved.

The reason is structural. `_build_bounds_user_defined_ration` boxes each feed within +/-10%
of the agent's percentage, and every cost coefficient is positive, so `min sum(c*x)` is
minimized at the all-lower-bounds corner whenever the NASEM constraints are already
satisfied there — and that corner does not depend on `c`. Prices only steer the mix when a
nutrition constraint binds hard enough to force some feed up.

Two consequences worth holding onto:

* **No confound.** RuFaS is not quietly doing part of the agent's job, so any
  price-responsiveness measured in a result is attributable to the policy.
* **The profit landscape is close to a linear program with time-varying coefficients.**
  Quantities are fixed by the ration, milk is nearly ration-insensitive, so
  `profit ~ milk_revenue - sum(price_i(t) * qty_i(ration))`. The optimal ration is roughly
  "lean on whatever is cheapest this month", which a fixed ration provably cannot track —
  but neither is it hard to find. A greedy price-following heuristic must therefore be one
  of the baselines, or beating CMA-ES will prove less than it appears to.

So `PriceApplier` still writes to the live `Feed` objects rather than re-pricing outputs
after the fact: it keeps the cost side of the reward as RuFaS's own reported economics, and
it stays correct if a future scenario's constraints do bind.

Three paths, all fixed at construction so an episode is reproducible from its seed:

* `StaticPricePath` — the scenario's own constant prices. The regression control: it must
  reproduce pre-price-lever results exactly.
* `SyntheticPricePath` — correlated mean-reverting processes. Training draws a fresh path
  per episode, so the policy sees unlimited price histories and cannot memorize one.
* `RealPricePath` — a resolved price CSV, for held-out evaluation on the real 2013-2019
  series the scenario's dates already span.

On the scenario's configured prices, for context: they are placeholders, and bad ones.
Corn grain sits at $0.50/kg DM (~3x its real cost) while every forage sits at $0.01/kg DM
(~10-20x too cheap). The 36% profit gain the CMA-ES baseline reports is substantially an
artifact of that — the optimizer dumps corn grain and piles into $0.01 corn silage. Real
prices are a correctness fix as much as a reformulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

#: Underlying commodities. Feed prices co-move because they are made of these, so the
#: process is defined on the commodities and feeds inherit; 5 correlated factors rather
#: than 10 loosely-coupled ones.
COMMODITIES: tuple[str, ...] = ("milk", "corn", "forage", "protein", "mineral")

#: Which commodity drives each of the scenario's feeds, by RuFaS feed ID. Names come from
#: the NASEM feed library (`NASEM_Comp_with_TDN.csv`).
FEED_COMMODITY: dict[int, str] = {
    23: "protein",   # Blood meal, low dRUP
    44: "corn",      # Corn grain dry, medium grind
    50: "corn",      # Corn silage, immature — priced off the corn crop
    95: "forage",    # Grass lg mixt, grass hay, mtr
    104: "forage",   # Grass lg mixt, legume slg
    110: "forage",   # Legume silage, mid-maturity
    202: "milk",     # Whole milk (fed to calves) — moves with the milk market
    216: "protein",  # Calf starter 18CP high starch
    301: "mineral",  # Farm ES Mineral Mix
    302: "protein",  # Farm ES Midwest BP Blend (byproducts)
}

#: A feed the map does not name is driven by corn, the dominant energy commodity. Chosen
#: over raising so an unfamiliar scenario still runs; `unmapped_feeds()` reports them.
DEFAULT_COMMODITY = "corn"

#: Plausible long-run price levels, $/kg **dry matter**, replacing the scenario's
#: placeholders. PROVISIONAL — round US figures a dairy nutritionist would recognize, not
#: a calibration; Stage 2 replaces them with values derived from the USDA series by
#: `scripts/build_prices.py`. They are here because the scenario's own numbers are not
#: merely imprecise but qualitatively wrong: corn grain at $0.50/kg DM (~3x real) against
#: every forage at $0.01/kg DM (~10-20x too cheap) makes "feed nothing but corn silage"
#: optimal, which is an artifact of the input file rather than a finding about dairy
#: farming.
#:
#: Note what changes: the forage/concentrate spread collapses from ~50x to under 2x, so
#: the profit optimum stops being an extreme corner and starts being a real trade-off.
REALISTIC_PRICES: dict[int, float] = {
    23: 1.10,   # Blood meal — high-protein, expensive
    44: 0.20,   # Corn grain dry (~$4.50/bu, 88% DM)
    50: 0.14,   # Corn silage (~$50/ton as-fed at 35% DM)
    95: 0.17,   # Grass hay (~$150/ton as-fed, 88% DM)
    104: 0.15,  # Grass-legume silage / haylage
    110: 0.18,  # Legume (alfalfa) silage
    202: 3.60,  # Whole milk fed to calves — 12.5% DM, so dear on a DM basis
    216: 0.60,  # Calf starter, 18% CP
    301: 1.20,  # Mineral/vitamin premix
    302: 0.25,  # Midwest byproduct blend (DDGS, soy hulls)
}
#: Milk paid to the farm, $/kg as sold (~$20/cwt). Matches `DEFAULT_MILK_PRICE`.
REALISTIC_MILK_PRICE = 0.45

#: Monthly AR(1) persistence and stationary standard deviation of log price, per
#: commodity. PROVISIONAL — order-of-magnitude values reflecting that milk and grain are
#: highly volatile and minerals are not. Stage 2 recalibrates these against the real
#: 2013-2019 series; `scripts/calibrate_prices.py` will overwrite them.
OU_PARAMS: dict[str, tuple[float, float]] = {
    # commodity: (phi, stationary log sd)
    "milk": (0.85, 0.16),
    "corn": (0.90, 0.20),
    "forage": (0.92, 0.15),
    "protein": (0.88, 0.18),
    "mineral": (0.95, 0.05),
}

#: Correlation of the monthly log-price innovations. Grain, forage and protein move
#: together (shared crop-year and energy-cost drivers); milk is only weakly coupled to
#: its own input costs; minerals are close to independent. PROVISIONAL, as above.
CORRELATION: dict[tuple[str, str], float] = {
    ("milk", "corn"): 0.20,
    ("milk", "forage"): 0.15,
    ("milk", "protein"): 0.15,
    ("milk", "mineral"): 0.05,
    ("corn", "forage"): 0.50,
    ("corn", "protein"): 0.60,
    ("corn", "mineral"): 0.10,
    ("forage", "protein"): 0.35,
    ("forage", "mineral"): 0.10,
    ("protein", "mineral"): 0.10,
}


@dataclass(frozen=True)
class PriceVector:
    """Prices in force for one month.

    `milk` is $/kg of milk as sold. `feeds` is $/kg of feed **dry matter**, keyed by RuFaS
    feed ID — the same basis `Feed.purchase_cost` uses, so it can be written straight onto
    a live `Feed`.
    """

    milk: float
    feeds: Mapping[int, float]

    def feed_price(self, rufas_id: int) -> float | None:
        return self.feeds.get(rufas_id)


class PricePath(Protocol):
    """Prices as a function of months elapsed since the simulation start."""

    def at(self, month_index: int) -> PriceVector: ...

    @property
    def reference(self) -> PriceVector:
        """Long-run mean prices. The scale the observation normalizes against."""
        ...


def commodity_of(rufas_id: int) -> str:
    return FEED_COMMODITY.get(rufas_id, DEFAULT_COMMODITY)


def correlation_matrix(commodities: tuple[str, ...] = COMMODITIES) -> np.ndarray:
    """Symmetric correlation matrix over `commodities`, built from `CORRELATION`."""
    n = len(commodities)
    m = np.eye(n)
    index = {c: i for i, c in enumerate(commodities)}
    for (a, b), rho in CORRELATION.items():
        if a in index and b in index:
            m[index[a], index[b]] = m[index[b], index[a]] = rho
    return m


class StaticPricePath:
    """Constant prices — the scenario exactly as configured.

    The regression control. An episode run under this path must reproduce results from
    before the price lever existed, which is what `test_prices.py` and
    `scripts/spike_prices.py` check.
    """

    def __init__(self, milk: float, feeds: Mapping[int, float]) -> None:
        self._vector = PriceVector(milk=float(milk), feeds=dict(feeds))

    def at(self, month_index: int) -> PriceVector:
        return self._vector

    @property
    def reference(self) -> PriceVector:
        return self._vector


class SyntheticPricePath:
    """Correlated mean-reverting prices, generated once per episode from a seed.

    Each commodity's log price follows a stationary AR(1) about its long-run mean:

        x[t] = phi * x[t-1] + sqrt(1 - phi^2) * sd * e[t],   e ~ N(0, CORRELATION)

    which has stationary standard deviation exactly `sd`, so the volatility parameters
    mean what they say regardless of `phi`. Prices are `mean * exp(x - sd^2/2)`; the
    Ito correction keeps the *expected* price equal to the long-run mean rather than
    `mean * exp(sd^2/2)`, so `reference` is unbiased and `volatility_scale=0` collapses
    to `StaticPricePath` exactly. That collapse is the module's main invariant test.

    Long-run means default to the scenario's own configured prices, so this path adds
    volatility and correlation *without* silently also changing price levels — level
    changes are Stage 2's job, carried by real data rather than constants in here.
    """

    def __init__(
        self,
        milk: float,
        feeds: Mapping[int, float],
        n_months: int,
        seed: int | None = None,
        volatility_scale: float = 1.0,
        commodities: tuple[str, ...] = COMMODITIES,
    ) -> None:
        if n_months < 1:
            raise ValueError(f"n_months must be >= 1, got {n_months}")
        if volatility_scale < 0:
            raise ValueError(f"volatility_scale must be >= 0, got {volatility_scale}")

        self._milk_mean = float(milk)
        self._feed_means = dict(feeds)
        self._commodities = commodities
        self._n_months = int(n_months)
        self.volatility_scale = float(volatility_scale)
        self.seed = seed

        self._log_deviation = self._simulate(seed)  # (n_months, n_commodities)
        self._cache: dict[int, PriceVector] = {}

    def _simulate(self, seed: int | None) -> np.ndarray:
        n_c = len(self._commodities)
        phi = np.array([OU_PARAMS[c][0] for c in self._commodities])
        sd = np.array([OU_PARAMS[c][1] for c in self._commodities]) * self.volatility_scale

        if self.volatility_scale == 0.0:
            return np.zeros((self._n_months, n_c))

        # Cholesky both correlates the innovations and asserts the matrix is a valid
        # correlation matrix — a non-PSD edit to CORRELATION fails loudly right here.
        chol = np.linalg.cholesky(correlation_matrix(self._commodities))
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((self._n_months, n_c)) @ chol.T

        x = np.empty((self._n_months, n_c))
        x[0] = sd * shocks[0]  # start drawn from the stationary distribution
        for t in range(1, self._n_months):
            x[t] = phi * x[t - 1] + np.sqrt(1.0 - phi**2) * sd * shocks[t]
        # Ito correction, so E[exp(x)] == 1 and the long-run mean price is unbiased.
        return x - 0.5 * sd**2

    def _multiplier(self, month_index: int) -> dict[str, float]:
        t = int(np.clip(month_index, 0, self._n_months - 1))
        return {c: float(np.exp(self._log_deviation[t, i]))
                for i, c in enumerate(self._commodities)}

    def at(self, month_index: int) -> PriceVector:
        key = int(np.clip(month_index, 0, self._n_months - 1))
        if key not in self._cache:
            mult = self._multiplier(key)
            self._cache[key] = PriceVector(
                milk=self._milk_mean * mult["milk"],
                feeds={fid: base * mult[commodity_of(fid)]
                       for fid, base in self._feed_means.items()},
            )
        return self._cache[key]

    @property
    def reference(self) -> PriceVector:
        return PriceVector(milk=self._milk_mean, feeds=dict(self._feed_means))

    def unmapped_feeds(self) -> tuple[int, ...]:
        """Scenario feeds falling back to `DEFAULT_COMMODITY` — none, as configured."""
        return tuple(sorted(f for f in self._feed_means if f not in FEED_COMMODITY))


class RealPricePath:
    """Prices read from a resolved CSV — held-out evaluation on real market history.

    The CSV is fully resolved: one row per month, prices already converted to $/kg and to
    a dry-matter basis, one column per scenario feed.

        month,milk,feed_23,feed_44,feed_50,...
        2013-01,0.4123,1.0850,0.1902,0.0781,...

    Deliberately *not* computed in this module. Turning USDA series (all-milk $/cwt, corn
    $/bu, alfalfa and other hay $/ton, soybean meal $/ton) into $/kg DM per feed involves
    unit conversions, dry-matter fractions and a feed-to-commodity basis that all deserve
    to be reviewable data rather than constants buried in code. `scripts/build_prices.py`
    (Stage 2) writes this file; this class only reads it.

    Rows are consumed in file order as months 0, 1, 2, ...; an index past the end clamps
    to the final row, matching `SyntheticPricePath`.
    """

    def __init__(self, csv_path: str | Path) -> None:
        import csv

        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"No price CSV at {self.path}. Real price paths are built by "
                "scripts/build_prices.py from the USDA series; until that has been run, "
                "use price_process='synthetic' or 'static'."
            )

        with self.path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"Price CSV {self.path} has no rows")

        feed_columns = {c: int(c.removeprefix("feed_"))
                        for c in rows[0] if c.startswith("feed_")}
        if not feed_columns:
            raise ValueError(f"Price CSV {self.path} defines no feed_<id> columns")

        self.months: list[str] = [r.get("month", str(i)) for i, r in enumerate(rows)]
        self._vectors = [
            PriceVector(
                milk=float(r["milk"]),
                feeds={fid: float(r[col]) for col, fid in feed_columns.items()},
            )
            for r in rows
        ]
        # Mean over the whole series, so observation normalization does not leak the
        # current month — it is a property of the series, not of where we are in it.
        self._reference = PriceVector(
            milk=float(np.mean([v.milk for v in self._vectors])),
            feeds={fid: float(np.mean([v.feeds[fid] for v in self._vectors]))
                   for fid in feed_columns.values()},
        )

    def at(self, month_index: int) -> PriceVector:
        return self._vectors[int(np.clip(month_index, 0, len(self._vectors) - 1))]

    @property
    def reference(self) -> PriceVector:
        return self._reference

    def __len__(self) -> int:
        return len(self._vectors)


#: How many months back the momentum feature looks. One quarter: long enough to carry a
#: trend past monthly noise, short enough to still be news at a monthly decision.
MOMENTUM_LAG = 3


class PriceFeatures:
    """The price block of the observation.

    Without this the policy cannot see prices and the whole reformulation is pointless —
    it would be a fixed-ration problem again, just with a noisier reward.

    Per price (milk, then each feed in ascending ID order), two numbers:

    * `log(p_t / reference)` — where this price sits against its own long-run mean. Scale-
      free, so a $1.00/kg mineral and a $0.01/kg silage contribute comparably, and the
      policy learns "forage is dear this month" rather than memorizing dollar levels.
    * `log(p_t / p_{t-3})` — which way it is moving. Prices are mean-reverting, so a price
      that is high *and* rising says something different from one that is high and falling.

    Ratios against a reference the *path* defines, never against the current month, so no
    information about the future leaks into the observation.
    """

    def __init__(self, feed_ids: tuple[int, ...], momentum_lag: int = MOMENTUM_LAG) -> None:
        self.feed_ids = tuple(feed_ids)
        self.momentum_lag = int(momentum_lag)

    @property
    def size(self) -> int:
        return 2 * (1 + len(self.feed_ids))

    def names(self) -> list[str]:
        keys = ["milk"] + [f"feed_{f}" for f in self.feed_ids]
        return ([f"price.{k}.vs_mean" for k in keys]
                + [f"price.{k}.momentum_{self.momentum_lag}m" for k in keys])

    def extract(self, path: PricePath, month_index: int) -> np.ndarray:
        current = path.at(month_index)
        # Clamped at 0, so early months compare against the first month and read as flat
        # momentum rather than as a fabricated trend.
        past = path.at(max(month_index - self.momentum_lag, 0))
        reference = path.reference

        def ratio(now: float, base: float) -> float:
            if base <= 0 or now <= 0:
                return 0.0
            return float(np.log(now / base))

        vs_mean = [ratio(current.milk, reference.milk)]
        momentum = [ratio(current.milk, past.milk)]
        for fid in self.feed_ids:
            vs_mean.append(ratio(current.feeds.get(fid, 0.0), reference.feeds.get(fid, 0.0)))
            momentum.append(ratio(current.feeds.get(fid, 0.0), past.feeds.get(fid, 0.0)))
        return np.asarray(vs_mean + momentum, dtype=np.float32)


def build_price_features(spec, config) -> PriceFeatures | None:
    """The price block for `config`, or None when prices are not observed.

    Shared by `RufasEnv` (which must size `observation_space` before any episode exists)
    and `Episode` (which fills the block), so the two cannot disagree about the layout.

    `config.observe_prices` defaults to None, meaning *auto*: observe prices exactly when
    they vary. That default exists to close a footgun — a varying price process that the
    policy cannot see makes the reward depend on hidden state, quietly breaking the Markov
    property and producing a policy that looks like it failed to learn when really it was
    never shown the information.
    """
    observe = getattr(config, "observe_prices", None)
    if observe is None:
        observe = config.price_process != "static"
    if not observe:
        return None
    return PriceFeatures(spec.feed_ids)


def scenario_prices(spec, levels: str = "configured") -> tuple[float, dict[int, float]]:
    """Long-run milk and feed price levels for a scenario.

    `levels="configured"` reads RuFaS's own `purchased_feed_cost` — use it to reproduce or
    regression-test existing results. `levels="realistic"` substitutes `REALISTIC_PRICES`
    for the scenario's placeholders, which is what any result meant to say something about
    dairy farming should run on.

    Milk price is not a RuFaS input in either case — RuFaS has no milk market — so it is
    supplied externally.
    """
    from .rewarders import DEFAULT_MILK_PRICE

    if levels == "configured":
        return DEFAULT_MILK_PRICE, dict(spec.feed_prices)
    if levels == "realistic":
        configured = dict(spec.feed_prices)
        # Any feed the table does not name keeps its configured price, so an unfamiliar
        # scenario degrades to partial substitution rather than silently losing a feed.
        return REALISTIC_MILK_PRICE, {
            fid: REALISTIC_PRICES.get(fid, price) for fid, price in configured.items()
        }
    raise ValueError(f"Unknown price levels {levels!r}; expected 'configured' or 'realistic'")


def unpriced_feeds(spec) -> tuple[int, ...]:
    """Scenario feeds `REALISTIC_PRICES` does not cover — none, as configured."""
    return tuple(f for f in spec.feed_ids if f not in REALISTIC_PRICES)


def make_price_path(
    spec,
    process: str = "static",
    n_months: int | None = None,
    seed: int | None = None,
    levels: str = "configured",
    **kwargs,
) -> PricePath:
    """Build the price path an episode runs under.

    `process` is one of `static`, `synthetic`, `real`; `levels` is `configured` or
    `realistic` (see `scenario_prices`). Extra kwargs go to the path class
    (`volatility_scale` for synthetic, `csv_path` for real).
    """
    milk, feeds = scenario_prices(spec, levels)

    if process == "static":
        return StaticPricePath(milk=milk, feeds=feeds)
    if process == "synthetic":
        if n_months is None:
            n_months = spec.n_boundaries("monthly")
        return SyntheticPricePath(
            milk=milk, feeds=feeds, n_months=n_months, seed=seed, **kwargs
        )
    if process == "real":
        csv_path = kwargs.pop("csv_path", None)
        if csv_path is None:
            raise ValueError("price_process='real' requires price_kwargs={'csv_path': ...}")
        if kwargs:
            raise TypeError(f"Unexpected kwargs for the real price path: {sorted(kwargs)}")
        return RealPricePath(csv_path)

    raise ValueError(
        f"Unknown price process {process!r}; expected 'static', 'synthetic' or 'real'"
    )
