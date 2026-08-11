"""The greedy price-following baseline."""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl.baselines import GreedyPriceRation, tune_sharpness
from rufas_rl.prices import PriceVector
from rufas_rl.tests.test_implementers import make_spec

# make_spec numbers feeds globally from 100, so ration 1 holds 102..108.
PRICES = PriceVector(
    milk=0.45,
    feeds={100: 1.0, 101: 2.0,
           102: 0.10, 103: 0.90, 104: 0.20, 105: 0.80, 106: 0.30, 107: 0.70, 108: 0.40,
           109: 0.5, 110: 0.5, 111: 0.5, 112: 0.5, 113: 0.5, 114: 0.5, 115: 0.5,
           116: 0.5, 117: 0.5, 118: 0.5, 119: 0.5, 120: 0.5, 121: 0.5, 122: 0.5},
)


@pytest.fixture
def greedy():
    return GreedyPriceRation(make_spec(), sharpness=1.0)


def test_cheapest_feed_gets_the_largest_share(greedy):
    ration = greedy.action(PRICES)
    shares = greedy.implementer.decode(ration)[1]  # the 7-feed ration, feeds 102..108
    assert np.argmax(shares) == 0          # feed 102 at $0.10 is cheapest
    assert np.argmin(shares) == 1          # feed 103 at $0.90 is dearest
    assert sum(shares) == pytest.approx(100.0)


def test_shares_are_monotone_in_price(greedy):
    """A cheaper feed must never receive less than a dearer one."""
    costs = [PRICES.feeds[f] for f in make_spec().ration_feed_ids[1]]
    shares = greedy.implementer.decode(greedy.action(PRICES))[1]
    by_cost = [s for _, s in sorted(zip(costs, shares))]
    assert by_cost == sorted(by_cost, reverse=True)


def test_zero_sharpness_is_an_even_split():
    greedy = GreedyPriceRation(make_spec(), sharpness=0.0)
    shares = greedy.implementer.decode(greedy.action(PRICES))[1]
    assert np.allclose(shares, 100.0 / 7)


def test_sharpness_increases_concentration():
    mild = GreedyPriceRation(make_spec(), sharpness=0.3)
    keen = GreedyPriceRation(make_spec(), sharpness=1.0)
    a = max(mild.implementer.decode(mild.action(PRICES))[1])
    b = max(keen.implementer.decode(keen.action(PRICES))[1])
    assert a < b


def test_policy_responds_when_prices_change(greedy):
    """The whole point: a different price vector must produce a different ration."""
    flipped = PriceVector(
        milk=0.45,
        feeds={**PRICES.feeds, 102: 0.90, 103: 0.10},  # swap cheapest and dearest
    )
    before = greedy.implementer.decode(greedy.action(PRICES))[1]
    after = greedy.implementer.decode(greedy.action(flipped))[1]
    assert np.argmax(after) == 1
    assert not np.allclose(before, after)


def test_action_is_inside_the_box(greedy):
    action = greedy.action(PRICES)
    assert action.shape == (greedy.implementer.size,)
    assert np.all(np.abs(action) <= 1.0 + 1e-9)


def test_two_feed_ration_still_ranks(greedy):
    """The calf ration has only two feeds; the rank mapping must not divide by zero."""
    shares = greedy.implementer.decode(greedy.action(PRICES))[0]
    assert shares[0] > shares[1]  # feed 100 at $1.00 beats feed 101 at $2.00
    assert sum(shares) == pytest.approx(100.0)


def test_invalid_sharpness_rejected():
    with pytest.raises(ValueError, match="sharpness"):
        GreedyPriceRation(make_spec(), sharpness=1.5)


def test_tune_sharpness_picks_the_best():
    out = tune_sharpness(lambda s: -((s - 0.5) ** 2), sharpnesses=(0.0, 0.5, 1.0))
    assert out["best_sharpness"] == 0.5
    assert set(out["sweep"]) == {0.0, 0.5, 1.0}
