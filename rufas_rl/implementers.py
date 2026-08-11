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

#: Temperature on the softmax. One scalar used to control two things at once, which is
#: why it caused trouble: for a 7-feed ration it sets both the *ceiling* on any one feed
#: (`e^s / (e^s + 6e^-s)`) and the *floor* under every other one (`e^-s / (...)`).
#:
#: It was cut 5.0 -> 1.5 because concentrated rations crash RuFaS's manure chemistry, and
#: a crash on any step ends the episode. But the cut is what put the CMA-ES optimum out of
#: PPO's reach, and the diagnosis of *why* was wrong. At 1.5 the ceiling is 77% — high
#: enough. The binding constraint was the floor: 3.8% each, so the five feeds a
#: concentrated ration wants to switch off still occupy >=19% of it. CMA-ES's winner put
#: 69.6%/28.2% on two feeds and ~2.2% across the other five, and no value of this scalar
#: expresses that shape.
#:
#: So the two jobs are now separated: the temperature goes back up to free the floor, and
#: the crash region is excluded directly by `MAX_INCLUSION` below. At 5.0 the floor is
#: ~0.005%, so "switch this feed off" is finally in the action space.
LOGIT_SCALE = 5.0

#: Per-feed inclusion ceilings as a fraction of ration dry matter, by RuFaS feed ID.
#: `DEFAULT_MAX_INCLUSION` covers anything unlisted.
#:
#: Measured by `scripts/probe_crash_boundary.py` (1-year episodes, one feed pushed to a
#: target share with the rest of its ration spread evenly), because the inherited comment
#: — "feeds #50 and #301 above ~60-80%" — turned out to be wrong about both the feeds and
#: the threshold:
#:
#:     feed  23 (blood meal)    survived 50/70/90/97%
#:     feed  50 (corn silage)   survived 50/70/90/97%
#:     feed 301 (mineral mix)   survived 50/70/90%, CRASHED at 97%
#:
#: Only the mineral premix breaks RuFaS, and only at 97%. So the 1.5 temperature — a 77%
#: ceiling — was guarding against a boundary roughly 20 points above where it sat, at the
#: cost of a 3.8% floor under every feed. 0.85 keeps a wide margin under the one measured
#: crash, sits far above the ~70% any profit-optimal ration has wanted, and is generous by
#: dairy standards (real rations rarely exceed ~60% of one ingredient).
#:
#: This bounds single-feed concentration only; the probe did not sweep combinations, so
#: the per-episode crash rate under a trained policy is worth watching. Both optimizers
#: face the same ceilings, so the comparison stays fair even where one binds — but CMA-ES
#: results predating these caps are not comparable and must be re-run.
MAX_INCLUSION: dict[int, float] = {}
DEFAULT_MAX_INCLUSION = 0.85


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))  # shift for numerical stability
    return e / e.sum()


def cap_shares(shares: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Project `shares` onto the simplex under per-element ceilings `caps`.

    Water-filling: pin whatever breaches its ceiling, redistribute the freed mass across
    the rest in proportion to their original weights, and repeat until nothing breaches.
    Each pass pins at least one element, so it terminates in at most `len(shares)` passes.

    Proportional redistribution (rather than an L2 projection) keeps the *relative*
    preferences among the uncapped feeds exactly as the policy expressed them, which is
    what makes the gradient the policy sees well behaved.
    """
    shares = np.asarray(shares, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.float64)
    if caps.sum() < 1.0 - 1e-9:
        raise ValueError(
            f"Inclusion caps sum to {caps.sum():.4f} < 1; no valid ration satisfies them."
        )

    out = shares.copy()
    pinned = np.zeros(len(shares), dtype=bool)
    for _ in range(len(shares)):
        breach = (out > caps + 1e-12) & ~pinned
        if not breach.any():
            break
        pinned |= breach
        out[pinned] = caps[pinned]
        free = ~pinned
        remaining = 1.0 - caps[pinned].sum()
        weight = shares[free].sum()
        if not free.any() or remaining <= 0:
            break
        # If every free feed sits at zero weight there is nothing to scale, so spread the
        # remainder evenly rather than dividing by zero.
        out[free] = (shares[free] / weight * remaining if weight > 0
                     else remaining / free.sum())
    return out


class RationImplementer:
    """Maps a flat action vector to per-group ration percentages.

    A note on timing: RuFaS re-formulates rations every `formulation_interval` days
    (default 30), so at monthly cadence an action applied at a pause is picked up by the
    next formulation. Applying one at a finer cadence is legal but will not be seen until
    that interval elapses.
    """

    lever = "rations"

    def __init__(
        self,
        spec: ScenarioSpec,
        logit_scale: float | None = None,
        max_inclusion: dict[int, float] | None = None,
    ) -> None:
        self.spec = spec
        self.logit_scale = LOGIT_SCALE if logit_scale is None else float(logit_scale)
        self.max_inclusion = MAX_INCLUSION if max_inclusion is None else dict(max_inclusion)
        self.ration_sizes = spec.ration_sizes
        self.ration_groups = spec.ration_groups
        self._offsets: list[tuple[int, int]] = []
        start = 0
        for size in self.ration_sizes:
            self._offsets.append((start, start + size))
            start += size
        self.size = start
        # One cap vector per ration, aligned with that ration's feeds. Built once here so
        # decode() stays allocation-light on the hot path.
        self._caps = [
            np.array([self.max_inclusion.get(fid, DEFAULT_MAX_INCLUSION) for fid in ids],
                     dtype=np.float64)
            for ids in spec.ration_feed_ids
        ] if spec.ration_feed_ids else [
            np.full(size, DEFAULT_MAX_INCLUSION) for size in self.ration_sizes
        ]

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
        for (start, end), caps in zip(self._offsets, self._caps):
            shares = softmax(action[start:end] * self.logit_scale)
            if (caps < 1.0).any():
                shares = cap_shares(shares, caps)
            out.append((shares * 100.0).tolist())
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
            parts.append(np.clip(logits / self.logit_scale, -ACTION_LIMIT, ACTION_LIMIT))
        return np.concatenate(parts).astype(np.float32)
