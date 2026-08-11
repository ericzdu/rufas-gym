"""Turning the paused farm into the agent's state vector.

**Why we read live engine objects instead of `variables_pool`.** The pool does accumulate
during a run, so it is readable at a pause — but most of the soil variables are written by
*annual* reporters (`send_soil_annual_variables`). At a monthly boundary in July, soil
nitrogen read from the pool is whatever was reported the previous January: up to eleven
months stale, and blind to every action the agent has taken since. That quietly breaks the
Markov property. The live object graph hanging off the paused engine is the actual current
state, so that is what we read.

Reading is pure inspection — we never write through these references — so this does not
touch RuFaS's dynamics.

The vector is fixed-length and laid out as:

    [ calendar (3) | per-field (20 x n_fields) | herd (11) ]

Values are divided by rough magnitude scales and clipped, so everything lands in roughly
[-10, 10] for the policy network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# (attribute, scale) pairs. Scales are order-of-magnitude normalizers taken from a real
# freestall run — they only need to put features on comparable footing, not be exact.
SOIL_FEATURES: tuple[tuple[str, float], ...] = (
    ("profile_nitrates_total", 100.0),
    ("profile_ammonium_total", 10.0),
    ("profile_active_organic_nitrogen_total", 500.0),
    ("profile_stable_organic_nitrogen_total", 20000.0),
    ("profile_fresh_organic_nitrogen_total", 100.0),
    ("profile_carbon_total", 200000.0),
    ("profile_soil_water_content", 500.0),
    ("available_phosphorus_pool", 50.0),
    ("annual_runoff_nitrates_total", 100.0),
    ("annual_runoff_total", 500.0),
)

CROP_FEATURES: tuple[tuple[str, float], ...] = (
    ("biomass", 20000.0),
    ("above_ground_biomass", 15000.0),
    ("leaf_area_index", 6.0),
    ("accumulated_heat_units", 2000.0),
    ("heat_fraction", 1.0),
    ("nitrogen", 300.0),
)

STRESS_FEATURES: tuple[str, ...] = (
    "nitrogen_stress",
    "phosphorus_stress",
    "temp_stress",
    "water_stress",
)

HERD_FEATURES: tuple[tuple[str, float], ...] = (
    ("cow_num", 200.0),
    ("milking_cow_num", 200.0),
    ("dry_cow_num", 50.0),
    ("calf_num", 50.0),
    ("heiferI_num", 100.0),
    ("heiferII_num", 100.0),
    ("avg_days_in_milk", 300.0),
    ("avg_cow_body_weight", 800.0),
    ("avg_parity_num", 5.0),
    ("preg_cow_num", 200.0),
    ("daily_milk_production", 5000.0),
)

N_CALENDAR = 3
N_PER_FIELD = len(SOIL_FEATURES) + len(CROP_FEATURES) + len(STRESS_FEATURES)
N_HERD = len(HERD_FEATURES)

OBS_CLIP = 10.0


def _get(obj, name: str, default: float = 0.0) -> float:
    """Read a numeric attribute, tolerating absence and None.

    Deliberately forgiving: RuFaS leaves plenty of fields as None outside a growing
    season, and a scenario with a different module set may not expose every attribute.
    An observation should degrade to zeros, not crash an episode mid-training.
    """
    value = getattr(obj, name, None)
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


@dataclass(frozen=True)
class ObservationLayout:
    """Where each block sits in the vector — handy for debugging and for `info`."""

    n_fields: int
    #: Width of the trailing price block, 0 when prices are not observed. Appended last so
    #: every existing index keeps its meaning.
    n_price: int = 0
    price_names: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return N_CALENDAR + N_PER_FIELD * self.n_fields + N_HERD + self.n_price

    def names(self) -> list[str]:
        names = ["month_sin", "month_cos", "year_progress"]
        for i in range(self.n_fields):
            names += [f"field{i}.soil.{a}" for a, _ in SOIL_FEATURES]
            names += [f"field{i}.crop.{a}" for a, _ in CROP_FEATURES]
            names += [f"field{i}.stress.{a}" for a in STRESS_FEATURES]
        names += [f"herd.{a}" for a, _ in HERD_FEATURES]
        names += list(self.price_names)
        return names


class FarmObserver:
    """Extracts a fixed-length observation from a paused `SimulationEngine`.

    `price_features` is optional: pass a `prices.PriceFeatures` to append the price block,
    leave it None for the physical-state-only observation. The block goes at the end, so a
    policy or a test written against the old layout still reads the same indices.
    """

    def __init__(self, n_fields: int, price_features=None) -> None:
        self.price_features = price_features
        self.layout = ObservationLayout(
            n_fields=n_fields,
            n_price=price_features.size if price_features is not None else 0,
            price_names=tuple(price_features.names()) if price_features is not None else (),
        )

    @property
    def size(self) -> int:
        return self.layout.size

    def observe(self, engine, price_block: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros(self.size, dtype=np.float32)
        i = 0

        # --- calendar: cyclical month + how far through the horizon we are ---
        date = engine.time.current_date
        out[i] = math.sin(2 * math.pi * date.month / 12.0)
        out[i + 1] = math.cos(2 * math.pi * date.month / 12.0)
        total_days = max(_get(engine.time, "simulation_length_days", 1.0), 1.0)
        out[i + 2] = _get(engine.time, "simulation_day") / total_days
        i += N_CALENDAR

        # --- fields ---
        # A scenario's simulation_type decides which managers exist at all: an
        # animals_only engine has no field_manager, a field_only engine has no
        # herd_manager. Missing blocks stay zeroed rather than raising.
        field_manager = getattr(engine, "field_manager", None)
        fields = list(getattr(field_manager, "fields", []) or [])
        for f_idx in range(self.layout.n_fields):
            # A scenario with fewer fields than declared leaves this block zeroed.
            field = fields[f_idx] if f_idx < len(fields) else None

            soil_data = getattr(getattr(field, "soil", None), "data", None)
            for attr, scale in SOIL_FEATURES:
                out[i] = _get(soil_data, attr) / scale
                i += 1

            # Aggregate over whatever crops are on the field: sums for stocks, maxima
            # for the intensive measures. An empty field contributes zeros.
            crops = list(getattr(field, "crops", []) or []) if field is not None else []
            crop_data = [c.data for c in crops if getattr(c, "data", None) is not None]
            for attr, scale in CROP_FEATURES:
                if not crop_data:
                    value = 0.0
                elif attr in ("biomass", "above_ground_biomass", "nitrogen"):
                    value = sum(_get(d, attr) for d in crop_data)
                else:
                    value = max(_get(d, attr) for d in crop_data)
                out[i] = value / scale
                i += 1

            # Stresses are already 0-1 fractions. Average across growing crops only —
            # a dormant or absent crop is not "unstressed", it is simply not growing.
            growing = [c for c in crops if getattr(getattr(c, "data", None), "is_growing", False)]
            for attr in STRESS_FEATURES:
                if growing:
                    out[i] = sum(
                        _get(getattr(c, "growth_constraints", None), attr) for c in growing
                    ) / len(growing)
                i += 1

        # --- herd ---
        stats = getattr(getattr(engine, "herd_manager", None), "herd_statistics", None)
        for attr, scale in HERD_FEATURES:
            out[i] = _get(stats, attr) / scale
            i += 1

        # --- prices ---
        # Already log-ratios, so no scaling: they are dimensionless and O(0.1) by
        # construction. A missing block stays zeroed, which reads as "every price is at
        # its long-run mean and flat" — the right degradation.
        if self.layout.n_price:
            if price_block is not None:
                block = np.asarray(price_block, dtype=np.float32).reshape(-1)
                if block.size != self.layout.n_price:
                    raise ValueError(
                        f"Price block has {block.size} features, expected "
                        f"{self.layout.n_price}"
                    )
                out[i:i + self.layout.n_price] = block
            i += self.layout.n_price

        assert i == self.size, f"observation layout mismatch: wrote {i}, expected {self.size}"
        return np.clip(out, -OBS_CLIP, OBS_CLIP, out=out)
