"""Shared fixtures.

Most of the package is pure data transformation, so the fast tests run against a stub
engine rather than a real simulation. The stub mimics only the attribute *paths* the
observer and rewarder read — which is precisely what those tests are checking.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest


def make_engine(
    *,
    day: int = 31,
    date: dt.datetime | None = None,
    n_fields: int = 2,
    milk: float = 3000.0,
    runoff_n: float = 10.0,
    growing: bool = True,
) -> SimpleNamespace:
    """A minimal stand-in for a paused `SimulationEngine`."""

    def field(idx: int) -> SimpleNamespace:
        soil_data = SimpleNamespace(
            profile_nitrates_total=50.0 + idx,
            profile_ammonium_total=1.0,
            profile_active_organic_nitrogen_total=200.0,
            profile_stable_organic_nitrogen_total=10000.0,
            profile_fresh_organic_nitrogen_total=9.0,
            profile_carbon_total=150000.0,
            profile_soil_water_content=210.0,
            available_phosphorus_pool=13.0,
            annual_runoff_nitrates_total=runoff_n,
            annual_runoff_total=200.0,
        )
        crop_data = SimpleNamespace(
            biomass=7000.0,
            above_ground_biomass=5000.0,
            leaf_area_index=4.5,
            accumulated_heat_units=485.0,
            heat_fraction=0.42,
            nitrogen=100.0,
            is_growing=growing,
        )
        crop = SimpleNamespace(
            data=crop_data,
            growth_constraints=SimpleNamespace(
                nitrogen_stress=0.2,
                phosphorus_stress=0.1,
                temp_stress=0.05,
                water_stress=0.0,
            ),
        )
        return SimpleNamespace(
            soil=SimpleNamespace(data=soil_data),
            crops=[crop],
            field_data=SimpleNamespace(field_size=10.0),
        )

    return SimpleNamespace(
        time=SimpleNamespace(
            current_date=date or dt.datetime(2013, 2, 1),
            simulation_day=day,
            simulation_length_days=2556,
            current_simulation_year=1,
        ),
        field_manager=SimpleNamespace(fields=[field(i) for i in range(n_fields)]),
        herd_manager=SimpleNamespace(
            herd_statistics=SimpleNamespace(
                cow_num=101,
                milking_cow_num=90,
                dry_cow_num=11,
                calf_num=10,
                heiferI_num=46,
                heiferII_num=40,
                avg_days_in_milk=147.0,
                avg_cow_body_weight=644.0,
                avg_parity_num=2.2,
                preg_cow_num=66,
                daily_milk_production=milk,
            )
        ),
    )


@pytest.fixture
def engine():
    return make_engine()
