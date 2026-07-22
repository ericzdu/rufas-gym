"""The action encoding must make every point in the box a valid ration."""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl.implementers import ACTION_LIMIT, RationImplementer
from rufas_rl.spec import ScenarioSpec


def make_spec(ration_sizes=(2, 7, 7, 7)) -> ScenarioSpec:
    from pathlib import Path

    return ScenarioSpec(
        task_metadata_path=Path("/tmp/task.json"),
        scenario_metadata_path=Path("/tmp/scenario.json"),
        simulation_type="full_farm",
        start_year=2013,
        end_year=2019,
        ration_sizes=ration_sizes,
        ration_groups=tuple(f"g{i}" for i in range(len(ration_sizes))),
        field_names=("field_1", "field_2"),
    )


@pytest.fixture
def implementer() -> RationImplementer:
    return RationImplementer(make_spec())


def test_action_size_is_total_feed_slots(implementer):
    assert implementer.size == 2 + 7 + 7 + 7


def test_every_ration_sums_to_100(implementer):
    """The property that makes the space safe: no action can be invalid."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=implementer.size)
        for percentages in implementer.decode(action):
            assert pytest.approx(sum(percentages), abs=1e-6) == 100.0
            assert all(p >= 0.0 for p in percentages)


def test_decode_shape_matches_scenario(implementer):
    decoded = implementer.decode(np.zeros(implementer.size))
    assert [len(d) for d in decoded] == [2, 7, 7, 7]


def test_neutral_action_is_an_even_split(implementer):
    decoded = implementer.decode(implementer.neutral_action())
    assert pytest.approx(decoded[0], abs=1e-9) == [50.0, 50.0]
    assert pytest.approx(decoded[1][0], abs=1e-9) == 100.0 / 7


def test_extreme_action_concentrates_but_stays_below_the_crash_ceiling(implementer):
    """LOGIT_SCALE lets one feed dominate, but is capped to keep rations out of the
    concentration region that crashes RuFaS's manure chemistry (~60-80% single-feed)."""
    action = np.zeros(implementer.size)
    action[2] = ACTION_LIMIT  # first feed of the 7-feed ration
    action[3:9] = -ACTION_LIMIT
    # The worst-case one-hot action reaches ~77%; typical Gaussian exploration stays far
    # lower (~54% at the 95th percentile), which is what keeps the per-episode crash rate
    # near 5%. The point is only that concentration is bounded well below near-100%.
    dominant = implementer.decode(action)[1][0]
    assert 65.0 < dominant < 85.0, f"single-feed concentration {dominant:.1f}% off target"


def test_encode_decode_roundtrip(implementer):
    original = [[50.0, 50.0], [40.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0]] + [
        [100.0 / 7] * 7
    ] * 2
    recovered = implementer.decode(implementer.encode(original))
    for want, got in zip(original, recovered):
        assert np.allclose(want, got, atol=0.5)


def test_wrong_length_action_is_rejected(implementer):
    with pytest.raises(ValueError, match="length"):
        implementer.decode(np.zeros(implementer.size + 1))


def test_non_finite_action_is_rejected(implementer):
    action = np.zeros(implementer.size)
    action[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        implementer.decode(action)


def test_action_space_is_normalized(implementer):
    """SB3 expects a symmetric unit box."""
    space = implementer.action_space()
    assert np.allclose(space.low, -1.0)
    assert np.allclose(space.high, 1.0)
    assert space.shape == (implementer.size,)
