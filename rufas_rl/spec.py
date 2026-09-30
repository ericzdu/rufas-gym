"""Static description of a scenario, read straight off its JSON inputs.

Gymnasium wants `action_space` and `observation_space` to exist at `__init__`, before
any `reset()`. Both depend on the scenario's shape — how many rations there are and how
many feeds each contains, how many fields, how long the run is. All of that is plain
JSON, so we read it directly instead of booting RuFaS: no InputManager, no simulation,
no subprocess. It costs a few file reads.

The traversal mirrors how RuFaS itself resolves a run:

    task_manager_metadata.json  ->  files.tasks.path
      tasks.json                ->  tasks[i].metadata_file_path
        scenario metadata.json  ->  files.{feed,config,field_*}.path
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .bootstrap import resolve


def _read_json(path: str | Path) -> dict:
    return json.loads(resolve(path).read_text())


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything the env needs to declare its spaces, known before any run."""

    task_metadata_path: Path
    scenario_metadata_path: Path
    simulation_type: str
    start_year: int
    end_year: int
    #: number of feeds in each ration, in file order — the action vector's segments
    ration_sizes: tuple[int, ...]
    #: `animal_combination` of each ration, same order
    ration_groups: tuple[str, ...]
    #: RuFaS feed IDs in each ration, in the same order as that ration's action segment.
    #: Lets per-feed logic (inclusion caps, price-ranking policies) line up with the
    #: action vector without re-reading the feed file.
    ration_feed_ids: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    field_names: tuple[str, ...] = field(default_factory=tuple)
    #: RuFaS feed IDs available to the scenario, ascending — the keys a price path is
    #: written against. From the feed file's `feeds[].feed_type`.
    feed_ids: tuple[int, ...] = field(default_factory=tuple)
    #: Configured `purchased_feed_cost` as `(feed_id, $/kg dry matter)` pairs, ascending.
    #: These are the long-run means a synthetic price path varies around, and the
    #: constants a static path reproduces. Pairs rather than a dict so the frozen
    #: dataclass stays hashable; `dict(spec.feed_prices)` is the usual read.
    feed_prices: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    #: Crude protein (% of dry matter) of each feed in each ration, aligned with
    #: `ration_feed_ids`; `None` where the composition table has no entry. From the
    #: scenario's NASEM composition table — the same one RuFaS formulates against. Needed
    #: for the protein floor in `implementers.py`.
    ration_feed_cp: tuple[tuple[float | None, ...], ...] = field(default_factory=tuple)

    @property
    def n_action_slots(self) -> int:
        return sum(self.ration_sizes)

    @property
    def n_fields(self) -> int:
        return len(self.field_names)

    @property
    def n_years(self) -> int:
        return self.end_year - self.start_year + 1

    def n_boundaries(self, cadence: str) -> int:
        """Upper bound on decision points, used to size the episode horizon."""
        return self.n_years * (12 if cadence == "monthly" else 1)


def _read_crude_protein(path: str | Path) -> dict[int, float]:
    """`rufas_id -> CP (% DM)` from a NASEM composition CSV."""
    with resolve(path).open(newline="") as fh:
        return {int(row["rufas_id"]): float(row["CP"])
                for row in csv.DictReader(fh) if row.get("CP") not in (None, "")}


def _parse_year(date_str: str) -> int:
    """RuFaS config dates are 'YYYY:DDD' (year:julian-day)."""
    return int(str(date_str).split(":")[0])


def load_spec(
    task_metadata_path: str | Path = "input/task_manager_metadata.json",
    task_index: int = 0,
) -> ScenarioSpec:
    """Read a scenario's static shape. `task_index` selects which task in the file."""
    task_meta_path = resolve(task_metadata_path)
    task_meta = _read_json(task_meta_path)
    tasks = _read_json(task_meta["files"]["tasks"]["path"])["tasks"]
    if not tasks:
        raise ValueError(f"No tasks defined in {task_meta_path}")
    task = tasks[task_index]

    scenario_meta_path = resolve(task["metadata_file_path"])
    files = _read_json(scenario_meta_path)["files"]

    config = _read_json(files["config"]["path"])
    feed = _read_json(files["feed"]["path"])

    rations = feed.get("rations", [])
    ration_sizes = tuple(len(r["feeds"]) for r in rations)
    ration_groups = tuple(r.get("animal_combination", f"group_{i}") for i, r in enumerate(rations))
    ration_feed_ids = tuple(
        tuple(int(f["feed_type"]) for f in r["feeds"]) for r in rations
    )

    # The purchasable feed list is separate from the rations that draw on it.
    prices = {int(f["feed_type"]): float(f["purchased_feed_cost"])
              for f in feed.get("feeds", [])}
    feed_ids = tuple(sorted(prices))
    feed_prices = tuple((i, prices[i]) for i in feed_ids)

    # Scenarios without a NASEM table (field-only) simply get no CP data.
    cp = _read_crude_protein(files["NASEM_Comp"]["path"]) if "NASEM_Comp" in files else {}
    ration_feed_cp = tuple(tuple(cp.get(fid) for fid in ids) for ids in ration_feed_ids)

    # Fields are declared as `field_1`, `field_2`, ... keys in the scenario metadata.
    field_names = tuple(sorted(k for k in files if k.startswith("field_")))

    return ScenarioSpec(
        task_metadata_path=task_meta_path,
        scenario_metadata_path=scenario_meta_path,
        simulation_type=config.get("simulation_type", "full_farm"),
        start_year=_parse_year(config["start_date"]),
        end_year=_parse_year(config["end_date"]),
        ration_sizes=ration_sizes,
        ration_groups=ration_groups,
        ration_feed_ids=ration_feed_ids,
        field_names=field_names,
        feed_ids=feed_ids,
        feed_prices=feed_prices,
        ration_feed_cp=ration_feed_cp,
    )
