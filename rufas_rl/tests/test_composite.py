"""The composite action space must split cleanly into per-lever fragments."""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl.composite import MULT_HIGH, MULT_LOW, CompositeImplementer

from .test_implementers import make_spec


def build(levers):
    return CompositeImplementer(make_spec(), levers)


def test_size_is_sum_of_lever_widths():
    ci = build(("rations", "fertilizer", "manure"))
    # 23 ration slots + 2 fields fertilizer + 2 fields manure
    assert ci.size == 23 + 2 + 2
    assert ci.action_space().shape == (27,)


def test_rations_only_matches_bare_ration_implementer():
    ci = build(("rations",))
    assert ci.size == 23
    assert "fertilizer" not in ci.decode(np.zeros(ci.size))


def test_segments_are_contiguous_and_ordered():
    ci = build(("rations", "fertilizer", "manure"))
    assert ci._segments["rations"] == (0, 23)
    assert ci._segments["fertilizer"] == (23, 25)
    assert ci._segments["manure"] == (25, 27)


def test_field_lever_bounds_map_to_multiplier_range():
    ci = build(("rations", "fertilizer"))
    hi = ci.decode(np.ones(ci.size))
    lo = ci.decode(-np.ones(ci.size))
    assert hi["fertilizer"] == [MULT_HIGH, MULT_HIGH]
    assert lo["fertilizer"] == [MULT_LOW, MULT_LOW]


def test_neutral_action_is_baseline_rate_and_even_ration():
    ci = build(("rations", "fertilizer", "manure"))
    dec = ci.decode(ci.neutral_action())
    assert dec["fertilizer"] == pytest.approx([1.0, 1.0])
    assert dec["manure"] == pytest.approx([1.0, 1.0])
    assert sum(dec["rations"][0]) == pytest.approx(100.0)


def test_decode_rejects_wrong_length():
    ci = build(("rations", "fertilizer"))
    with pytest.raises(ValueError, match="length"):
        ci.decode(np.zeros(ci.size + 1))


def test_field_multipliers_are_one_per_field():
    ci = build(("rations", "fertilizer", "manure"))
    dec = ci.decode(ci.action_space().sample())
    assert len(dec["fertilizer"]) == 2
    assert len(dec["manure"]) == 2
    assert all(MULT_LOW <= m <= MULT_HIGH for m in dec["fertilizer"])
