"""Axis B — getting a decision into a farm that is already running.

Injecting into `InputManager.pool` mid-run does nothing. Every module copies its config
*out* of the pool during construction and never looks back: rations end up on
`RationManager` class attributes, field schedules become pre-built event lists on each
`Field`. `formulate_rations` does not re-read the pool.

So an applier re-invokes **RuFaS's own construction-time setup function** against the
live objects, with new numbers. It calls the same code path the simulator would have
called had the scenario been configured that way from the start — which is exactly why
the result is equivalent, and why this is plumbing rather than a change to the science.

Appliers run on the driver thread while the worker is blocked at a pause, so there is no
race: only one side is ever running.

**Only the ration applier is proven.** `scripts/spike_axisb_rations.py` and
`scripts/spike_pause_and_mutate.py` both show 0 physical `variables_pool` differences
against construction injection. Crop/fertilizer/manure appliers must clear the same
equivalence bar before they are added here — until then those levers are set once per
episode at construction.
"""

from __future__ import annotations

from typing import Protocol


class Applier(Protocol):
    lever: str

    def apply(self, decoded) -> None: ...


class RationApplier:
    """Rewrites the herd's rations mid-run.

    PROVEN equivalent to construction-time injection (0 of 2,701 physical keys differ).

    Reads the scenario's feed blob fresh from the InputManager each time — `get_data`
    returns a deepcopy, and the pool is never mutated — so applying successive actions
    stays idempotent rather than compounding.
    """

    lever = "rations"

    def apply(self, decoded: list[list[float]]) -> None:
        from RUFAS.biophysical.animal.ration.ration_manager import RationManager
        from RUFAS.input_manager import InputManager

        feed = InputManager().get_data("feed")
        rations = feed.get("rations", [])
        if len(decoded) != len(rations):
            raise ValueError(
                f"Decoded {len(decoded)} rations but the scenario defines {len(rations)}"
            )

        for ration, percentages in zip(rations, decoded):
            feeds = ration["feeds"]
            if len(feeds) != len(percentages):
                raise ValueError(
                    f"Ration {ration.get('animal_combination')!r} has {len(feeds)} feeds "
                    f"but {len(percentages)} percentages were decoded"
                )
            for entry, pct in zip(feeds, percentages):
                entry["ration_percentage"] = float(pct)

        # Both setters, in this order — the tolerance is derived from the same blob and
        # the manager will reject a ration whose percentages drift outside it.
        RationManager.set_user_defined_rations(feed)
        RationManager.set_user_defined_ration_tolerance(feed)


class _FieldScheduleApplier:
    """Base for field levers that scale a dated schedule (fertilizer, manure).

    The move, proven physically identical to construction injection by
    `scripts/spike_axisb_fertilizer.py` (0 of 2,702 keys differ): mutate the schedule blob
    in the InputManager pool, re-invoke RuFaS's own `_setup_*` function so it rebuilds the
    events exactly as construction would, and reassign the result onto the live field.
    `Field._filter_events` then keeps only the future-dated events, so applying at a year
    boundary changes the rest of the horizon and leaves already-applied events alone.

    Scaling is always relative to the *baseline* schedule captured the first time the
    applier runs, so re-applying a multiplier each year does not compound.
    """

    lever: str = ""
    spec_key: str = ""  # field-config key naming this field's schedule blob
    scaled_fields: tuple[str, ...] = ("nitrogen_masses", "phosphorus_masses")

    def __init__(self) -> None:
        self._baseline: dict[str, dict] = {}  # schedule name -> original blob

    def _pool(self):
        from RUFAS.input_manager import InputManager

        return InputManager()._InputManager__pool

    def _rebuild(self, engine, schedule_name: str):  # pragma: no cover - overridden
        raise NotImplementedError

    def _reassign(self, field, rebuilt) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def apply(self, decoded: list[float]) -> None:
        """`decoded` is one multiplier per field, in field order."""
        engine = _current_engine()
        pool = self._pool()
        fields = list(engine.field_manager.fields)
        if len(decoded) != len(fields):
            raise ValueError(f"{self.lever}: got {len(decoded)} multipliers for {len(fields)} fields")

        for field, multiplier in zip(fields, decoded):
            field_name = field.field_data.name
            schedule_name = pool[field_name][self.spec_key]
            self._baseline.setdefault(schedule_name, _deepcopy(pool[schedule_name]))
            base = self._baseline[schedule_name]

            scaled = _deepcopy(base)
            for key in self.scaled_fields:
                if key in scaled:
                    scaled[key] = [v * float(multiplier) for v in base[key]]
            pool[schedule_name] = scaled

            self._reassign(field, self._rebuild(engine, schedule_name))


class FertilizerApplier(_FieldScheduleApplier):
    """Per-field nitrogen/phosphorus multiplier on the fertilizer schedule."""

    lever = "fertilizer"
    spec_key = "fertilizer_management_specification"

    def _rebuild(self, engine, schedule_name):
        from RUFAS.biophysical.field.manager.field_manager import FieldManager

        return FieldManager._setup_fertilizer_events(schedule_name)  # (mixes, events)

    def _reassign(self, field, rebuilt):
        mixes, events = rebuilt
        field.available_fertilizer_mixes = mixes
        field.fertilizer_events = events


class ManureApplier(_FieldScheduleApplier):
    """Per-field nitrogen/phosphorus multiplier on the manure schedule."""

    lever = "manure"
    spec_key = "manure_management_specification"

    def _rebuild(self, engine, schedule_name):
        from RUFAS.biophysical.field.manager.field_manager import FieldManager

        return FieldManager._setup_manure_events(schedule_name)  # events

    def _reassign(self, field, rebuilt):
        field.manure_events = rebuilt


class PriceApplier:
    """Reprices the farm's feeds mid-run.

    **Exogenous, not a lever.** Every other applier here carries an agent decision; this
    one carries the market. It is driven by the episode's `PricePath`, never by the
    action vector, and so is deliberately absent from `APPLIERS` and `get_applier` —
    `EnvConfig.levers` must not be able to name it.

    The mechanism is simpler than the other appliers and needs no rebuild step. RuFaS
    builds one `list[Feed]` at construction (`AvailableFeedsBuilder.setup_available_feeds`)
    and hands the *same list object* to the feed manager, the herd manager and every pen;
    `Feed` is a plain mutable dataclass. So writing `purchase_cost` on the engine's feeds
    is seen everywhere at once, with no reconstruction and no stale copies. Nothing
    rebuilds that list mid-run, so there is nothing to fight with.

    What repricing actually moves, as measured by `scripts/spike_prices.py`:

    * **What RuFaS reports — yes.** `FeedManager.purchase_feed` prices purchases at
      `Feed.purchase_cost` and writes `ration_interval_<id>_cost` to the output pool,
      which is exactly what the `profit` rewarder reads. The cost side of the reward
      stays RuFaS's own economics, now at the current month's prices.
    * **What RuFaS physically feeds — no, not in this scenario.** Ration formulation
      *is* a least-cost program (`RationOptimizer.objective` is `sum(amounts * prices)`,
      re-solved every 30 days within 10% of the agent's ration), but a 30x relative price
      swing moved the purchased kilograms by exactly zero. All cost coefficients are
      positive, so the program sits at its all-lower-bounds corner whenever the NASEM
      constraints are slack there, and that corner is price-independent.

    Writing to the live objects anyway is still the right call: it keeps the reward
    denominated in RuFaS's own reported dollars rather than a re-priced copy, and it
    remains correct for a scenario whose nutrition constraints do bind. See `prices.py`
    for what this means for the experiment — briefly, the agent's ration is the only
    price-responsive channel, which removes a confound and demands a greedy heuristic
    baseline.

    `on_farm_cost` is kept at RuFaS's own fixed ratio to the purchase price for
    consistency. It happens to be write-only in RuFaS today — nothing reads it — but
    letting the two drift would be a trap for whoever wires up home-grown feed valuation.
    """

    lever = "prices"

    def apply(self, prices) -> None:
        """`prices` is a `PriceVector`; feeds it does not name are left alone."""
        from RUFAS.data_structures.feed_storage_to_animal_connection import (
            ON_FARM_TO_PURCHASED_PRICE_RATIO,
        )

        engine = _current_engine()
        feeds = getattr(engine, "available_feeds", None)
        if not feeds:
            raise RuntimeError(
                "The paused engine exposes no `available_feeds`; prices cannot be "
                "applied. (A field-only scenario has no herd and no feeds — run the "
                "price lever only on a scenario that simulates animals.)"
            )

        applied = 0
        for feed in feeds:
            price = prices.feed_price(feed.rufas_id)
            if price is None:
                continue
            feed.purchase_cost = float(price)
            feed.on_farm_cost = float(price) * ON_FARM_TO_PURCHASED_PRICE_RATIO
            applied += 1

        if applied == 0:
            raise ValueError(
                f"No scenario feed matched the price vector's IDs {sorted(prices.feeds)}. "
                f"The engine offers {sorted(f.rufas_id for f in feeds)}."
            )


def _current_engine():
    """The live engine, set by the episode at each pause so appliers can reach it."""
    if _ENGINE[0] is None:
        raise RuntimeError("No live engine registered; call set_engine() at each pause")
    return _ENGINE[0]


# Field appliers need the live engine (to reach its fields); the ration applier does not
# (it drives RationManager class state). The episode publishes the current engine here at
# each pause rather than threading it through every apply() signature.
_ENGINE: list = [None]


def set_engine(engine) -> None:
    _ENGINE[0] = engine


def _deepcopy(obj):
    import copy

    return copy.deepcopy(obj)


APPLIERS: dict[str, Applier] = {
    "rations": RationApplier(),
    "fertilizer": FertilizerApplier(),
    "manure": ManureApplier(),
}


def get_applier(lever: str) -> Applier:
    # A fresh instance per call for the stateful field appliers (they cache a per-episode
    # baseline); the stateless ration applier can be shared.
    if lever == "rations":
        return APPLIERS["rations"]
    if lever == "fertilizer":
        return FertilizerApplier()
    if lever == "manure":
        return ManureApplier()
    raise ValueError(
        f"No applier for lever {lever!r}. Available: rations, fertilizer, manure. "
        "(crop rotation is a categorical lever, not yet wired.)"
    )
