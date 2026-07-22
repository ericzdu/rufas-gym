"""The observation must be well-formed for anything the simulator can hand us."""

from __future__ import annotations

import numpy as np

from rufas_rl.observers import OBS_CLIP, FarmObserver

from .conftest import make_engine


def test_size_matches_layout_and_names():
    observer = FarmObserver(n_fields=2)
    assert observer.observe(make_engine()).shape == (observer.size,)
    assert len(observer.layout.names()) == observer.size


def test_values_are_finite_and_inside_the_declared_space(engine):
    observer = FarmObserver(n_fields=2)
    obs = observer.observe(engine)
    assert np.all(np.isfinite(obs))
    assert np.all(np.abs(obs) <= OBS_CLIP)
    assert obs.dtype == np.float32


def test_state_change_is_visible():
    """An observation that ignored soil state would break the MDP silently."""
    observer = FarmObserver(n_fields=1)
    low = observer.observe(make_engine(n_fields=1, runoff_n=1.0))
    high = observer.observe(make_engine(n_fields=1, runoff_n=90.0))
    assert not np.allclose(low, high)


def test_missing_fields_are_zeroed_not_fatal():
    """Declaring more fields than the scenario has must degrade, not crash."""
    observer = FarmObserver(n_fields=4)
    obs = observer.observe(make_engine(n_fields=2))
    assert obs.shape == (observer.size,)
    assert np.all(np.isfinite(obs))


def test_field_only_engine_without_a_herd():
    """A field_only scenario has no herd_manager at all."""
    engine = make_engine()
    del engine.herd_manager
    obs = FarmObserver(n_fields=2).observe(engine)
    assert np.all(np.isfinite(obs))


def test_animals_only_engine_without_fields():
    engine = make_engine()
    del engine.field_manager
    obs = FarmObserver(n_fields=2).observe(engine)
    assert np.all(np.isfinite(obs))


def test_empty_field_contributes_zero_crop_features():
    """Bare ground between crops is common and must not produce NaN."""
    engine = make_engine(n_fields=1)
    engine.field_manager.fields[0].crops = []
    obs = FarmObserver(n_fields=1).observe(engine)
    assert np.all(np.isfinite(obs))


def test_stress_is_ignored_for_non_growing_crops():
    """A dormant crop is not 'unstressed' — its stresses must not be averaged in."""
    observer = FarmObserver(n_fields=1)
    names = observer.layout.names()
    idx = names.index("field0.stress.nitrogen_stress")
    growing = observer.observe(make_engine(n_fields=1, growing=True))
    dormant = observer.observe(make_engine(n_fields=1, growing=False))
    assert growing[idx] > 0.0
    assert dormant[idx] == 0.0


def test_calendar_features_track_the_date():
    import datetime as dt

    observer = FarmObserver(n_fields=1)
    jan = observer.observe(make_engine(n_fields=1, date=dt.datetime(2013, 1, 1)))
    jul = observer.observe(make_engine(n_fields=1, date=dt.datetime(2013, 7, 1)))
    assert not np.allclose(jan[:2], jul[:2])
