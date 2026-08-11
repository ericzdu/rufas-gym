"""Price-path tests.

These are pure-data tests — no RuFaS, no simulation. The claim that repricing actually
changes what the *simulator* does is a different claim and is checked by
`scripts/spike_prices.py`, which runs real episodes.

The load-bearing invariant here is the collapse: a synthetic path at zero volatility must
equal the static path exactly. That is what lets `price_process="synthetic"` be trusted as
a superset of the old behaviour rather than a rewrite of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl.prices import (
    COMMODITIES,
    FEED_COMMODITY,
    OU_PARAMS,
    PriceVector,
    RealPricePath,
    StaticPricePath,
    SyntheticPricePath,
    commodity_of,
    correlation_matrix,
    make_price_path,
)

MILK = 0.45
FEEDS = {23: 0.1, 44: 0.5, 50: 0.01, 95: 0.01, 104: 0.01, 110: 0.01,
         202: 0.1, 216: 0.1, 301: 1.0, 302: 0.1}


def test_static_path_is_constant():
    path = StaticPricePath(milk=MILK, feeds=FEEDS)
    assert path.at(0) == path.at(11) == path.at(500)
    assert path.at(0).milk == MILK
    assert path.at(0).feed_price(44) == 0.5
    assert path.reference == path.at(0)


def test_unknown_feed_price_is_none():
    """`PriceApplier` relies on `None` to mean 'leave this feed alone'."""
    assert StaticPricePath(milk=MILK, feeds=FEEDS).at(0).feed_price(9999) is None


def test_correlation_matrix_is_a_valid_correlation_matrix():
    m = correlation_matrix()
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)
    np.linalg.cholesky(m)  # raises if a CORRELATION edit made it non-PSD


def test_every_scenario_feed_is_mapped_to_a_commodity():
    """A silent fall-through to DEFAULT_COMMODITY would decorrelate a feed by accident."""
    unmapped = [f for f in FEEDS if f not in FEED_COMMODITY]
    assert unmapped == []
    assert set(FEED_COMMODITY.values()) <= set(COMMODITIES)
    assert set(OU_PARAMS) == set(COMMODITIES)


def test_zero_volatility_collapses_to_static():
    """The invariant that makes the price lever a superset of the old environment."""
    static = StaticPricePath(milk=MILK, feeds=FEEDS)
    synthetic = SyntheticPricePath(
        milk=MILK, feeds=FEEDS, n_months=24, seed=0, volatility_scale=0.0
    )
    for month in range(24):
        assert synthetic.at(month).milk == pytest.approx(static.at(month).milk)
        for fid in FEEDS:
            assert synthetic.at(month).feed_price(fid) == pytest.approx(
                static.at(month).feed_price(fid)
            )


def test_synthetic_path_is_seed_reproducible_and_seed_sensitive():
    a = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=24, seed=7)
    b = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=24, seed=7)
    c = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=24, seed=8)
    assert [a.at(t).milk for t in range(24)] == [b.at(t).milk for t in range(24)]
    assert [a.at(t).milk for t in range(24)] != [c.at(t).milk for t in range(24)]


def test_synthetic_path_actually_varies():
    """A path that does not move would silently reproduce the static experiment."""
    path = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=84, seed=0)
    milk = np.array([path.at(t).milk for t in range(84)])
    corn = np.array([path.at(t).feed_price(44) for t in range(84)])
    assert milk.std() / milk.mean() > 0.05
    assert corn.std() / corn.mean() > 0.05
    assert (milk > 0).all() and (corn > 0).all()


def test_feeds_sharing_a_commodity_move_together():
    """Corn grain and corn silage are the same crop; they must not drift apart."""
    path = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=84, seed=3)
    assert commodity_of(44) == commodity_of(50) == "corn"
    grain = np.array([path.at(t).feed_price(44) for t in range(84)])
    silage = np.array([path.at(t).feed_price(50) for t in range(84)])
    # Same multiplier applied to different bases, so the ratio is exactly constant.
    assert np.allclose(grain / silage, grain[0] / silage[0])


def test_long_run_mean_is_unbiased():
    """The Ito correction: E[price] is the configured mean, not mean*exp(sd^2/2)."""
    samples = [
        SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=120, seed=s)
        for s in range(60)
    ]
    milk = np.array([p.at(t).milk for p in samples for t in range(120)])
    assert milk.mean() == pytest.approx(MILK, rel=0.03)
    assert samples[0].reference.milk == MILK


def test_stationary_volatility_matches_the_parameter():
    paths = [SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=240, seed=s)
             for s in range(40)]
    logs = np.array([[np.log(p.at(t).milk) for t in range(240)] for p in paths])
    assert logs.std() == pytest.approx(OU_PARAMS["milk"][1], rel=0.15)


def test_month_index_clamps_past_the_end():
    path = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=12, seed=0)
    assert path.at(11).milk == path.at(99).milk
    assert path.at(0).milk == path.at(-5).milk


def test_volatility_scale_scales_volatility():
    quiet = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=240, seed=1,
                               volatility_scale=0.25)
    loud = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=240, seed=1)
    q = np.std([np.log(quiet.at(t).milk) for t in range(240)])
    l = np.std([np.log(loud.at(t).milk) for t in range(240)])
    assert q < l


def test_real_path_reads_a_resolved_csv(tmp_path):
    csv = tmp_path / "prices.csv"
    csv.write_text(
        "month,milk,feed_44,feed_50\n"
        "2013-01,0.40,0.20,0.08\n"
        "2013-02,0.50,0.30,0.10\n"
    )
    path = RealPricePath(csv)
    assert len(path) == 2
    assert path.at(0).milk == 0.40
    assert path.at(1).feed_price(44) == 0.30
    assert path.at(50).milk == 0.50  # clamps
    # Reference is the series mean, so normalization does not leak the current month.
    assert path.reference.milk == pytest.approx(0.45)
    assert path.reference.feed_price(50) == pytest.approx(0.09)


def test_real_path_missing_file_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_prices"):
        RealPricePath(tmp_path / "nope.csv")


def test_price_features_shape_and_names():
    from rufas_rl.prices import PriceFeatures

    pf = PriceFeatures(tuple(sorted(FEEDS)))
    assert pf.size == 2 * (1 + len(FEEDS))
    assert len(pf.names()) == pf.size
    assert pf.names()[0] == "price.milk.vs_mean"


def test_price_features_are_zero_on_a_static_path():
    """Constant prices sit exactly at their mean with no momentum."""
    from rufas_rl.prices import PriceFeatures

    pf = PriceFeatures(tuple(sorted(FEEDS)))
    feats = pf.extract(StaticPricePath(milk=MILK, feeds=FEEDS), 12)
    assert np.allclose(feats, 0.0)


def test_price_features_signal_a_dear_month():
    """A price above its long-run mean must read positive, and vice versa."""
    from rufas_rl.prices import PriceFeatures

    class _Spiky:
        reference = PriceVector(milk=0.45, feeds={44: 0.20})

        def at(self, t):
            return PriceVector(milk=0.90 if t == 5 else 0.45,
                               feeds={44: 0.10 if t == 5 else 0.20})

    pf = PriceFeatures((44,))
    dear = pf.extract(_Spiky(), 5)
    assert dear[0] == pytest.approx(np.log(2.0), rel=1e-5)   # milk vs mean
    assert dear[1] == pytest.approx(np.log(0.5), rel=1e-5)   # corn vs mean
    assert np.allclose(pf.extract(_Spiky(), 0), 0.0)


def test_price_feature_momentum_uses_the_lag_and_clamps_early():
    from rufas_rl.prices import PriceFeatures

    path = SyntheticPricePath(milk=MILK, feeds=FEEDS, n_months=24, seed=2)
    pf = PriceFeatures(tuple(sorted(FEEDS)), momentum_lag=3)
    n = 1 + len(FEEDS)
    expected = np.log(path.at(9).milk / path.at(6).milk)
    assert pf.extract(path, 9)[n] == pytest.approx(expected, rel=1e-5)
    # At month 0 there is no history, so momentum must read flat rather than invented.
    assert pf.extract(path, 0)[n] == pytest.approx(0.0)


def test_realistic_levels_fix_the_broken_forage_concentrate_spread():
    """The placeholder prices are what made 'feed only corn silage' optimal."""
    from rufas_rl.prices import REALISTIC_PRICES

    configured_spread = FEEDS[44] / FEEDS[50]      # corn grain vs corn silage
    realistic_spread = REALISTIC_PRICES[44] / REALISTIC_PRICES[50]
    assert configured_spread == pytest.approx(50.0)
    assert realistic_spread < 2.0


def test_realistic_levels_cover_every_scenario_feed():
    from rufas_rl.prices import REALISTIC_PRICES

    assert set(FEEDS) <= set(REALISTIC_PRICES)


class _Spec:
    feed_prices = FEEDS
    feed_ids = tuple(sorted(FEEDS))
    n_years = 7

    def n_boundaries(self, cadence):
        return 84


def test_make_price_path_dispatch():
    spec = _Spec()
    assert isinstance(make_price_path(spec, "static"), StaticPricePath)
    assert isinstance(make_price_path(spec, "synthetic", seed=0), SyntheticPricePath)
    with pytest.raises(ValueError, match="Unknown price process"):
        make_price_path(spec, "vibes")
    with pytest.raises(ValueError, match="csv_path"):
        make_price_path(spec, "real")


def test_make_price_path_uses_scenario_prices_as_means():
    """Synthetic must vary around the scenario's own prices, not invented levels."""
    path = make_price_path(_Spec(), "synthetic", seed=0)
    assert path.reference.feeds == FEEDS


def test_make_price_path_honours_realistic_levels():
    from rufas_rl.prices import REALISTIC_PRICES

    path = make_price_path(_Spec(), "synthetic", seed=0, levels="realistic")
    assert path.reference.feeds[50] == REALISTIC_PRICES[50]
    assert path.reference.feeds[44] == REALISTIC_PRICES[44]


def test_unknown_price_levels_rejected():
    with pytest.raises(ValueError, match="Unknown price levels"):
        make_price_path(_Spec(), "static", levels="made_up")


def test_build_price_features_auto_follows_the_process():
    """Auto must observe prices exactly when they vary — the anti-footgun default."""
    from rufas_rl.config import EnvConfig
    from rufas_rl.prices import build_price_features

    spec = _Spec()
    assert build_price_features(spec, EnvConfig(price_process="static")) is None
    assert build_price_features(spec, EnvConfig(price_process="synthetic")) is not None
    # Explicit override still wins in both directions.
    assert build_price_features(
        spec, EnvConfig(price_process="synthetic", observe_prices=False)
    ) is None
    assert build_price_features(
        spec, EnvConfig(price_process="static", observe_prices=True)
    ) is not None
