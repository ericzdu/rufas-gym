"""Scenario spec and config validation. These read RuFaS's inputs but run no simulation."""

from __future__ import annotations

import pytest

from rufas_rl.config import EnvConfig
from rufas_rl.spec import load_spec

# Every scenario shipped in RuFaS's input directory, spanning both simulation types.
SCENARIOS = [
    "input/task_manager_metadata.json",
    "input/kimberly_rotation_task_manager_metadata.json",
    "input/farm_es_task_manager_metadata.json",
]


def test_default_scenario_spec():
    spec = load_spec()
    assert spec.simulation_type == "full_farm"
    assert spec.n_years == 7
    assert spec.n_fields == 2
    assert spec.n_action_slots == sum(spec.ration_sizes)
    assert spec.n_boundaries("monthly") == 84
    assert spec.n_boundaries("yearly") == 7


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_spec_loads_without_running_anything(scenario):
    """The spaces must be derivable from JSON alone, before any episode exists."""
    spec = load_spec(scenario)
    assert spec.n_action_slots > 0
    assert spec.end_year >= spec.start_year
    assert spec.simulation_type


def test_config_accepts_wired_field_levers():
    """Rations, fertilizer and manure all have passing equivalence spikes."""
    cfg = EnvConfig(levers=("rations", "fertilizer", "manure"))
    assert cfg.levers == ("rations", "fertilizer", "manure")


def test_config_rejects_unwired_levers():
    """Crop rotation is a categorical lever, not yet wired."""
    with pytest.raises(ValueError, match="No proven applier"):
        EnvConfig(levers=("rations", "crop"))


def test_config_rejects_no_levers():
    with pytest.raises(ValueError, match="At least one lever"):
        EnvConfig(levers=())


def test_unknown_cadence_is_rejected():
    from rufas_rl.stepper import get_cadence

    with pytest.raises(ValueError, match="Unknown cadence"):
        get_cadence("fortnightly")


def test_cadence_predicates():
    import datetime as dt
    from types import SimpleNamespace

    from rufas_rl.stepper import get_cadence

    def at(date):
        return SimpleNamespace(time=SimpleNamespace(current_date=date))

    monthly, yearly = get_cadence("monthly"), get_cadence("yearly")
    assert monthly(at(dt.datetime(2013, 3, 1)))
    assert not monthly(at(dt.datetime(2013, 3, 2)))
    assert yearly(at(dt.datetime(2013, 1, 1)))
    assert not yearly(at(dt.datetime(2013, 3, 1)))
