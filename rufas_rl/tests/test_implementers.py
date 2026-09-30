"""The action encoding must make every point in the box a valid ration."""

from __future__ import annotations

import numpy as np
import pytest

from rufas_rl.implementers import ACTION_LIMIT, RationImplementer
from rufas_rl.spec import ScenarioSpec


def _sequential_feed_ids(ration_sizes):
    ids, n = [], 100
    for size in ration_sizes:
        ids.append(tuple(range(n, n + size)))
        n += size
    return tuple(ids)


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
        # Synthetic feed IDs numbered globally, so feed id == 100 + action index. Keeps
        # "cap feed X" and "set action slot i" unambiguous across rations.
        ration_feed_ids=_sequential_feed_ids(ration_sizes),
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


def test_a_feed_can_be_switched_off(implementer):
    """The floor, which is what the old LOGIT_SCALE=1.5 actually cost us.

    At 1.5 every feed had a 3.8% floor, so the five feeds a concentrated ration wants to
    drop still took >=19% of it — which is why CMA-ES's 69.6/28.2/~2.2%-across-five
    optimum was not expressible.

    Two feeds carry the ration here rather than one, because shares must sum to 100: with
    a single feed at the 85% inclusion cap the remaining 15% has to land somewhere, so
    "one feed at ~100%, the rest at ~0%" is arithmetically impossible by design. What has
    to be reachable is dropping a feed while *other* feeds absorb its mass.
    """
    action = np.zeros(implementer.size)
    action[2] = action[3] = ACTION_LIMIT  # two feeds share the ration
    action[4:9] = -ACTION_LIMIT           # the other five are switched off
    decoded = implementer.decode(action)[1]
    assert max(decoded[2:]) < 0.5, f"floor still {max(decoded[2:]):.2f}%"
    assert decoded[0] + decoded[1] > 99.0


def test_cma_es_shaped_ration_is_expressible(implementer):
    """The concrete shape the old action space could not reach, as a regression guard."""
    target = [[50.0, 50.0], [69.6, 28.2, 0.6, 0.6, 0.5, 0.3, 0.2]] + [[100.0 / 7] * 7] * 2
    recovered = implementer.decode(implementer.encode(target))[1]
    assert np.allclose(target[1], recovered, atol=1.0), recovered


def test_default_cap_bounds_every_feed(implementer):
    """No action, however extreme, may exceed DEFAULT_MAX_INCLUSION on any feed."""
    from rufas_rl.implementers import DEFAULT_MAX_INCLUSION

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        action = rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=implementer.size)
        for pcts in implementer.decode(action):
            worst = max(worst, max(pcts))
    assert worst <= DEFAULT_MAX_INCLUSION * 100.0 + 1e-6, f"reached {worst:.2f}%"


def test_default_cap_sits_under_the_measured_crash_boundary():
    """`scripts/probe_crash_boundary.py`: feed 301 crashed at 97%, nothing below 90%."""
    from rufas_rl.implementers import DEFAULT_MAX_INCLUSION

    assert DEFAULT_MAX_INCLUSION <= 0.90
    # ...but not so tight that it re-blocks the ~70% profit-optimal ration, which is the
    # mistake LOGIT_SCALE=1.5 made.
    assert DEFAULT_MAX_INCLUSION >= 0.75


def test_inclusion_caps_bound_concentration():
    """Concentration is bounded by MAX_INCLUSION now, not by the softmax temperature."""
    spec = make_spec()
    capped = RationImplementer(spec, max_inclusion={102: 0.55})
    action = np.zeros(capped.size)
    action[2] = ACTION_LIMIT  # feed 102 == action slot 2, first of the 7-feed ration
    action[3:9] = -ACTION_LIMIT
    decoded = capped.decode(action)[1]
    assert decoded[0] == pytest.approx(55.0, abs=1e-6)
    assert sum(decoded) == pytest.approx(100.0, abs=1e-6)


def test_cap_shares_redistributes_proportionally():
    from rufas_rl.implementers import cap_shares

    out = cap_shares(np.array([0.8, 0.1, 0.1]), np.array([0.5, 1.0, 1.0]))
    assert out[0] == pytest.approx(0.5)
    assert out.sum() == pytest.approx(1.0)
    assert out[1] == pytest.approx(out[2])  # equal inputs stay equal


def test_cap_shares_handles_cascading_breaches():
    """Pinning one element can push another over its own cap; it must re-run."""
    from rufas_rl.implementers import cap_shares

    out = cap_shares(np.array([0.7, 0.25, 0.05]), np.array([0.4, 0.3, 1.0]))
    assert out[0] == pytest.approx(0.4)
    assert out[1] == pytest.approx(0.3)
    assert out[2] == pytest.approx(0.3)
    assert out.sum() == pytest.approx(1.0)


def test_cap_shares_is_a_noop_when_nothing_breaches():
    from rufas_rl.implementers import cap_shares

    shares = np.array([0.5, 0.3, 0.2])
    assert np.allclose(cap_shares(shares, np.ones(3)), shares)


def test_infeasible_caps_are_rejected():
    from rufas_rl.implementers import cap_shares

    with pytest.raises(ValueError, match="no valid ration|sum to"):
        cap_shares(np.array([0.5, 0.5]), np.array([0.2, 0.2]))


def test_capped_rations_still_sum_to_100():
    spec = make_spec()
    capped = RationImplementer(spec, max_inclusion={102: 0.4, 103: 0.3})
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=capped.size)
        for pcts in capped.decode(action):
            assert pytest.approx(sum(pcts), abs=1e-6) == 100.0
            assert all(p >= 0.0 for p in pcts)


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


# -- protein floor -----------------------------------------------------------------

def make_protein_spec(cps=(95.0, 8.5, 7.9, 20.0, 20.5, 0.04, 26.2)) -> ScenarioSpec:
    """One lactating ration shaped like the freestall scenario's, with real CP values."""
    from dataclasses import replace

    spec = make_spec(ration_sizes=(len(cps),))
    return replace(spec, ration_groups=("lac_cow",), ration_feed_cp=(tuple(cps),))


def _cp(pcts, cps):
    return sum(p * c for p, c in zip(pcts, cps)) / 100.0


def test_no_action_decodes_below_the_protein_floor():
    """The property that stops the crash: every action is a >=14% CP lactating ration."""
    spec = make_protein_spec()
    impl = RationImplementer(spec)
    rng = np.random.default_rng(0)
    for _ in range(500):
        pcts = impl.decode(rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, impl.size))[0]
        assert _cp(pcts, spec.ration_feed_cp[0]) >= 14.0 - 1e-9
        assert sum(pcts) == pytest.approx(100.0, abs=1e-6)
        assert min(pcts) >= 0.0
        assert max(pcts) <= 85.0 + 1e-6  # the blend must not break the inclusion caps


def test_the_crashing_rations_are_lifted_exactly_to_the_floor():
    """Rations that crashed RuFaS in the diagnostic (85% mineral mix; 78% corn grain)."""
    spec = make_protein_spec()
    impl = RationImplementer(spec)
    for heavy in (5, 1):  # mineral mix, corn grain
        action = np.full(impl.size, -ACTION_LIMIT)
        action[heavy] = ACTION_LIMIT
        pcts = impl.decode(action)[0]
        assert _cp(pcts, spec.ration_feed_cp[0]) == pytest.approx(14.0, abs=1e-6)


def test_rations_above_the_floor_are_untouched():
    spec = make_protein_spec()
    floored = RationImplementer(spec)
    unfloored = RationImplementer(spec, min_crude_protein={})
    action = np.zeros(floored.size)  # even split: ~25% CP
    assert floored.decode(action) == unfloored.decode(action)


def test_the_configured_farm_ration_passes_the_floor():
    """The farm's own lactating ration (15.3% CP) must not be altered by the floor."""
    from rufas_rl.implementers import MIN_CRUDE_PROTEIN

    configured = [0.74, 15.31, 37.45, 0.37, 17.65, 3.27, 25.21]
    real_cp = [95.06, 8.514, 7.91, 20.044, 20.471, 0.0351, 26.1583]
    assert _cp(configured, real_cp) >= MIN_CRUDE_PROTEIN["lac_cow"]


def test_floor_without_crude_protein_data_is_rejected():
    spec = make_protein_spec(cps=(95.0, None, 7.9, 20.0, 20.5, 0.04, 26.2))
    with pytest.raises(ValueError, match="crude protein"):
        RationImplementer(spec)


def test_unreachable_floor_is_rejected():
    spec = make_protein_spec(cps=(9.0, 8.5, 7.9, 10.0, 10.5, 0.04, 12.0))
    with pytest.raises(ValueError, match="reaches"):
        RationImplementer(spec)
