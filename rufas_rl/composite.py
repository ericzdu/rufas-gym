"""Combining several levers into one action space.

The agent emits a single flat vector; this splits it into a ration segment (softmaxed to
per-group percentages) and one continuous multiplier per field for each field lever
(fertilizer, manure). The layout is fixed and introspectable so the env can build its
`Box` and the episode can route each segment to the right applier.

Action layout:  [ ration logits | fertilizer x n_fields | manure x n_fields ]

Field-lever multipliers are exposed on [-1, 1] (the SB3-friendly unit box) and mapped to a
positive scale via `MULT_LOW .. MULT_HIGH`, so 0 is roughly the baseline rate, +1 the
maximum, -1 zero application.
"""

from __future__ import annotations

import numpy as np

from .implementers import RationImplementer
from .spec import ScenarioSpec

FIELD_LEVERS = ("fertilizer", "manure")
MULT_LOW, MULT_HIGH = 0.0, 2.0  # multiplier range for a field lever


def _to_multiplier(raw: float) -> float:
    """Map a raw action value in [-1, 1] to a [MULT_LOW, MULT_HIGH] multiplier."""
    unit = (float(np.clip(raw, -1.0, 1.0)) + 1.0) / 2.0
    return MULT_LOW + unit * (MULT_HIGH - MULT_LOW)


def _from_multiplier(mult: float) -> float:
    unit = (mult - MULT_LOW) / (MULT_HIGH - MULT_LOW)
    return float(np.clip(unit * 2.0 - 1.0, -1.0, 1.0))


class CompositeImplementer:
    """Decodes one flat action into per-lever fragments."""

    def __init__(
        self,
        spec: ScenarioSpec,
        levers: tuple[str, ...],
        min_crude_protein: dict[str, float] | None = None,
    ) -> None:
        self.spec = spec
        self.levers = tuple(levers)
        self.ration = (RationImplementer(spec, min_crude_protein=min_crude_protein)
                       if "rations" in self.levers else None)
        self.n_fields = spec.n_fields
        self.field_levers = tuple(lv for lv in self.levers if lv in FIELD_LEVERS)

        # Fixed segment offsets into the action vector.
        self._segments: dict[str, tuple[int, int]] = {}
        i = 0
        if self.ration is not None:
            self._segments["rations"] = (i, i + self.ration.size)
            i += self.ration.size
        for lever in self.field_levers:
            self._segments[lever] = (i, i + self.n_fields)
            i += self.n_fields
        self.size = i

    def action_space(self):
        from gymnasium import spaces

        return spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def decode(self, action: np.ndarray) -> dict:
        """Return {lever: fragment}. Ration -> list[list[float]]; field -> list[float]."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.size:
            raise ValueError(f"Expected action of length {self.size}, got {action.size}")
        out: dict = {}
        if self.ration is not None:
            lo, hi = self._segments["rations"]
            out["rations"] = self.ration.decode(action[lo:hi])
        for lever in self.field_levers:
            lo, hi = self._segments[lever]
            out[lever] = [_to_multiplier(v) for v in action[lo:hi]]
        return out

    def neutral_action(self) -> np.ndarray:
        """Even ration split and baseline-rate field levers (multiplier ~1.0)."""
        a = np.zeros(self.size, dtype=np.float32)
        for lever in self.field_levers:
            lo, hi = self._segments[lever]
            a[lo:hi] = _from_multiplier(1.0)
        return a
