"""Reward is a flow between pauses, which makes the bookkeeping easy to get wrong."""

from __future__ import annotations

import pytest

from rufas_rl.rewarders import MilkMinusNitrogen, _annual_delta, make_rewarder

from .conftest import make_engine


def test_first_boundary_scores_nothing():
    """No interval has elapsed yet, so there is nothing to pay for."""
    rewarder = MilkMinusNitrogen()
    reward, info = rewarder.reward(make_engine(day=0, milk=3000.0))
    assert reward == 0.0
    assert info["milk_kg"] == 0.0


def test_milk_accumulates_over_the_interval():
    rewarder = MilkMinusNitrogen()
    rewarder.reward(make_engine(day=0, milk=3000.0, runoff_n=0.0))
    _, info = rewarder.reward(make_engine(day=31, milk=3000.0, runoff_n=0.0))
    assert info["days"] == 31
    assert info["milk_kg"] == pytest.approx(3000.0 * 31)


def test_milk_uses_the_trapezoid_of_both_endpoints():
    """The opening boundary reads a rate of 0 because herd stats are not populated."""
    rewarder = MilkMinusNitrogen()
    rewarder.reward(make_engine(day=0, milk=0.0, runoff_n=0.0))
    _, info = rewarder.reward(make_engine(day=31, milk=3000.0, runoff_n=0.0))
    assert info["milk_kg"] == pytest.approx(0.5 * 3000.0 * 31)
    assert info["milk_kg"] > 0.0, "the opening month must not be credited with zero milk"


def test_nitrogen_runoff_is_penalized():
    clean = MilkMinusNitrogen()
    clean.reward(make_engine(day=0, milk=3000.0, runoff_n=0.0))
    clean_reward, _ = clean.reward(make_engine(day=31, milk=3000.0, runoff_n=0.0))

    dirty = MilkMinusNitrogen()
    dirty.reward(make_engine(day=0, milk=3000.0, runoff_n=0.0))
    dirty_reward, info = dirty.reward(make_engine(day=31, milk=3000.0, runoff_n=50.0))

    assert dirty_reward < clean_reward
    assert info["nitrate_runoff_kg"] > 0


def test_runoff_is_scaled_by_field_size():
    """RuFaS reports runoff per hectare; the penalty should be on the field total."""
    rewarder = MilkMinusNitrogen()
    rewarder.reward(make_engine(day=0, n_fields=1, runoff_n=0.0))
    _, info = rewarder.reward(make_engine(day=31, n_fields=1, runoff_n=10.0))
    assert info["nitrate_runoff_kg"] == pytest.approx(10.0 * 10.0)  # 10 kg/ha x 10 ha


def test_annual_counter_reset_is_handled():
    """RuFaS zeroes annual totals each year; a drop means a reset, not negative runoff."""
    assert _annual_delta(30.0, 10.0) == 20.0  # normal accumulation
    assert _annual_delta(5.0, 90.0) == 5.0  # reset fired; 5 is all new
    assert _annual_delta(5.0, 90.0) >= 0.0


def test_reset_clears_history():
    rewarder = MilkMinusNitrogen()
    rewarder.reward(make_engine(day=0))
    rewarder.reward(make_engine(day=31))
    rewarder.reset()
    reward, info = rewarder.reward(make_engine(day=200))
    assert reward == 0.0 and info["days"] == 0.0


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown rewarder"):
        make_rewarder("does_not_exist")


# --- Profit -----------------------------------------------------------------
# The Profit rewarder reads RuFaS's feed cost from the output pool. To keep these unit
# tests fast and RuFaS-free we stub that one method; its correctness against real feed
# cost is covered by the slow end-to-end tests and the optimization experiment.
from rufas_rl.rewarders import Profit  # noqa: E402


def _profit_with_costs(costs):
    """A Profit rewarder whose feed-cost-to-date returns successive `costs`."""
    r = Profit(milk_price=0.45)
    seq = iter(costs)
    r._feed_cost_to_date = lambda: next(seq)  # type: ignore[method-assign]
    return r


def test_profit_is_revenue_minus_feed_cost():
    r = _profit_with_costs([0.0, 1000.0])
    r.reward(make_engine(day=0, milk=0.0))  # prime
    reward, info = r.reward(make_engine(day=31, milk=3000.0))
    expected_revenue = 0.5 * 3000.0 * 31 * 0.45
    assert info["milk_revenue"] == pytest.approx(expected_revenue)
    assert info["feed_cost"] == pytest.approx(1000.0)
    assert info["profit"] == pytest.approx(expected_revenue - 1000.0)


def test_profit_feed_cost_is_the_interval_delta():
    """Pool feed cost is cumulative; the reward must charge only the new spend."""
    r = _profit_with_costs([300.0, 800.0, 1500.0])
    r.reward(make_engine(day=0, milk=3000.0))  # prime: baseline 300, charges nothing
    _, first = r.reward(make_engine(day=31, milk=3000.0))
    _, second = r.reward(make_engine(day=62, milk=3000.0))
    assert first["feed_cost"] == pytest.approx(500.0)  # 300 -> 800
    assert second["feed_cost"] == pytest.approx(700.0)  # 800 -> 1500


def test_profit_first_boundary_charges_no_feed_cost():
    """The priming call sets the baseline; it must not bill accumulated spend."""
    r = _profit_with_costs([5000.0])
    _, info = r.reward(make_engine(day=0))
    assert info["feed_cost"] == pytest.approx(0.0)


def test_profit_preset_is_registered():
    assert isinstance(make_rewarder("profit"), Profit)
