"""Decoding an RL action vector into RuFaS lever settings.

The agent emits one flat `Box` vector (what SB3's policies produce). For rations that
vector is split into one segment per animal group and each segment is softmaxed onto a
simplex summing to 100 — so **every action in the box is a valid ration**, with no
clipping, rejection, or invalid-action masking. The feed *set* is fixed by the scenario;
the agent only re-weights it.

Decoding is deliberately separated from applying (`appliers.py`) because the same decoded
fragment has two consumers: the live appliers mid-episode, and a construction-time
`input_patch` for the replay oracle.
"""

from __future__ import annotations

import numpy as np

from .spec import ScenarioSpec

#: The action space is the symmetric unit box SB3's policies expect.
ACTION_LIMIT = 1.0
#: Actions are multiplied by this before the softmax, setting how concentrated a ration
#: the policy can express. Lowered from 5.0 to 1.5 after finding that heavy single-feed
#: rations crash RuFaS's manure chemistry (feeds #50 and #301 above ~60-80% drive the
#: cow's nitrogen balance negative). A crash on *any* step ends the episode, so per-step
#: crash risk compounds over ~24 monthly decisions: at 5.0 that was ~95% of episodes,
#: which flattened the reward and stalled learning. At 1.5 the per-step crash-prone rate
#: is ~0.2% (~5% per episode) — a clean signal to learn from. The cost is a concentration
#: ceiling of ~54% on any one feed, so the extreme concentrated-forage rations CMA-ES
#: reached (~70%) are out of reach; the policy can still shift heavily toward the cheap
#: forages and away from the dear concentrates, which is where the profit is.
LOGIT_SCALE = 1.5


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))  # shift for numerical stability
    return e / e.sum()


class RationImplementer:
    """Maps a flat action vector to per-group ration percentages.

    A note on timing: RuFaS re-formulates rations every `formulation_interval` days
    (default 30), so at monthly cadence an action applied at a pause is picked up by the
    next formulation. Applying one at a finer cadence is legal but will not be seen until
    that interval elapses.
    """

    lever = "rations"

    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.ration_sizes = spec.ration_sizes
        self.ration_groups = spec.ration_groups
        self._offsets: list[tuple[int, int]] = []
        start = 0
        for size in self.ration_sizes:
            self._offsets.append((start, start + size))
            start += size
        self.size = start

    def action_space(self):
        from gymnasium import spaces

        return spaces.Box(
            low=-ACTION_LIMIT,
            high=ACTION_LIMIT,
            shape=(self.size,),
            dtype=np.float32,
        )

    def decode(self, action: np.ndarray) -> list[list[float]]:
        """Action vector -> ration percentages, one inner list per ration."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.size:
            raise ValueError(f"Expected an action of length {self.size}, got {action.size}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values")

        out: list[list[float]] = []
        for start, end in self._offsets:
            out.append((softmax(action[start:end] * LOGIT_SCALE) * 100.0).tolist())
        return out

    def neutral_action(self) -> np.ndarray:
        """All-zero action — softmaxes to an even split within each ration.

        Useful as a control: it is a *valid* ration, not a no-op, so it does change the
        scenario relative to its configured baseline.
        """
        return np.zeros(self.size, dtype=np.float32)

    def encode(self, percentages: list[list[float]]) -> np.ndarray:
        """Approximate inverse of `decode` — log-percentages recover the simplex.

        Lets a scenario's configured ration be expressed as an action, e.g. to seed a
        baseline policy at the farm's actual current practice.
        """
        parts = []
        for pct in percentages:
            arr = np.asarray(pct, dtype=np.float64)
            arr = np.clip(arr, 1e-6, None)
            logits = np.log(arr)
            logits -= logits.mean()
            parts.append(np.clip(logits / LOGIT_SCALE, -ACTION_LIMIT, ACTION_LIMIT))
        return np.concatenate(parts).astype(np.float32)
