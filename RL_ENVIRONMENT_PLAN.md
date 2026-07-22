# Plan: A Dynamics-Preserving RL Environment for RuFaS — Sequential Whole-Farm Management

## Context

RuFaS is a validated whole-farm dairy simulator (sibling repo at `../RuFaS`). `../rufas-web` already
wraps it in a FastAPI + React interface, importing RuFaS in-process. Goal: expose RuFaS as a
**reinforcement-learning environment** where an agent makes **sequential management decisions** —
feed ratios, crop rotations, fertilizer (N), manure application — optimizing economic + environmental
objectives over a multi-year horizon. **Deliverable: a paper.**

Design precedent: [CyclesGym](https://github.com/kora-labs/cyclesgym) (local copy at `../cyclesgym`)
wraps the *Cycles* crop model as a Gym env. We reuse its component decomposition
(observer/implementer/rewarder/constrainer) and its first-class weather sampler.

Two constraints:

1. **RuFaS dynamics stay fixed.** The transition `P(s'|s,a)` and reward `R` are never altered — that
   is the validated substrate the paper rests on. Physics edits are forbidden. The primary stepping
   mechanism (threaded pause hook) edits zero RuFaS files; a minimal, flag-gated source patch
   (loop → resumable form) remains permitted as fallback, off-by-default byte-identical to released
   RuFaS. Every stepped mode is gated on reproducing a monolithic run exactly (V7/V10).
2. **Determinism is not required.** RuFaS carries mild stochasticity; we **average N runs per
   evaluation** for low-variance returns.

## Feasibility — verified against RuFaS source

Checked directly in `../RuFaS`:

- **Action injection:** `task_manager.py:1015` deep-merges `input_patch` into `input_manager.pool`
  before the engine is built; `input_patch` is hard-set to `None` for single runs
  (`task_manager.py:373`), so the harness supplies it via monkey-patch.
- **Merge semantics:** `deep_merge` (`util.py:340`) recurses into dicts and merges lists
  element-wise; `Utility.flatten_keys_to_nested_structure` (`util.py:65`) expands flat dot-path keys
  into the nested form.
- **In-process single-worker path:** `task_manager.py:176` sets the pool to `None` when
  `workers == 1`; `~:528` uses a plain `list(map(...))`.
- **Per-year boundary:** `_run_simulation_main_loop` (`simulation_engine.py:314`) is a
  `for simulation_year in range(...)` loop calling `_annual_simulation()` (`:320`) once per year — the
  only site the stepping patch touches. Carryover state lives in the engine object graph; `get_data()`
  runs only in `_setup_simulation_modules` (`:207-278`), not the daily loop, which mutates module
  attributes (`self.field_manager`, `self.herd`, `self.manure_manager`, `self.feed_manager`).
- **Singletons:** flush via `input_manager.py:1317` `flush_pool`, `output_manager.py:2202`
  `flush_pools`; state read via `output_manager.py:2181` `_get_flat_variables_pool`.
- **Monkey-patch precedent:** `rufas-web/api/pool_store.py:65-66,78` wraps
  `_handle_simulation_engine_run_tasks` and re-runs the deep-merge.
- **Config is cached at construction — mid-run pool injection is a NO-OP (verified 2026-07-08):**
  rations are read once at `HerdManager` construction (`herd_manager.py:191` `get_data("feed")`) into
  `RationManager` **class attributes** (`ration_manager.py` `set_user_defined_rations`);
  `formulate_rations` (`herd_manager.py:1621`) never re-reads the pool. Field ops are built for the
  whole horizon at `FieldManager` construction (`_setup_field`, `field_manager.py:150-200` →
  `_setup_crop_events` / `_setup_fertilizer_events` / `_setup_manure_events`); `annual_update_routine`
  (`:113`) only resets, never rebuilds. Hence `input_patch` works **only at construction**; per-period
  actions require live-object mutation (see Stepping strategy).
- **Live ration mutation ≡ construction injection (spike PASS, `scripts/spike_axisb_rations.py`):**
  re-invoking `RationManager.set_user_defined_rations`/`set_user_defined_ration_tolerance` mid-run is
  physically identical to injecting the same ration via `input_patch` at construction — 0 of 2,701
  physical `variables_pool` keys differ (941 differ vs. baseline, so the change bites).
- **In-process reuse degrades ~5×/run** (identical 2,845-key output, 27 s → 148 s) even after
  flushing both singleton pools — so the harness runs **one episode per fresh subprocess** (RuFaS
  itself uses `maxtasksperchild=1`).

> `api/*` files cited below live in **`../rufas-web/`**, not this repo.

---

## The decision problem: a sequential MDP

The farm carries state forward and past actions have delayed, coupled consequences. At each decision
period `t`:

- **State `s_t`** — carryover from `variables_pool`: per-field soil N/P/C and moisture, standing crop
  & residue, herd structure (counts by age & lactation stage, body condition), feed/manure storage,
  calendar/season.
- **Action `a_t`** — the upcoming period's decisions (feed ratios, crop, fertilizer N, manure),
  injected via `input_patch`.
- **Transition** — advance the simulator one period; the daily biophysical loop produces `s_{t+1}`.
- **Reward `r_t`** — that period's profit minus GHG/leaching penalties.
- **Return** — discounted sum over the multi-year horizon.

Actions couple across time: a legume fixes N this year, raising next year's corn yield;
manure/fertilizer over-application builds soil N that leaches years later but also soil carbon that
raises future yields; ration choices drive body condition → milk & reproduction → herd structure for
years.

### Decision cadence (semi-MDP)

The agent acts only at **decision events**; the simulator runs its daily loop autonomously between
them. The RL timestep is the management decision cadence, not RuFaS's daily tick. Cadence is a
configurable knob implemented as a boundary predicate in the stepper: **monthly (~30-day), aligned
with ration reformulation (`formulation_interval`, default 30), is the primary target** (decided
2026-07-08); **yearly** is the same predicate turned coarse, used for bring-up and ablations. Note
the lever split: rations are the only *mid-year* lever; crop/fertilizer/manure change at year
boundaries regardless of cadence.

### Paper positioning

- **Contribution:** framing whole-farm dairy management on a validated simulator as a sequential RL
  environment, learning a multi-objective long-horizon policy that captures cross-period trade-offs
  (N carryover, soil carbon, herd dynamics). Broader scope than CyclesGym (single field, corn): herd
  + crop + soil + manure + economics coupled. Plus a reusable harness.
- **Baselines:** (i) a greedy/myopic per-period optimizer and (ii) a static full-horizon schedule
  optimized by CMA-ES/Optuna.
- **Optional secondary analysis:** inject weather/price variation, report a "value of adaptation"
  (≈ Value of the Stochastic Solution, Birge & Louveaux).

**Defaults:** cadence **monthly (primary target)**, yearly as the bring-up/ablation knob; objectives
profit, profit−GHG, N-leaching (report all three); first levers feed ratios + crop rotation +
fertilizer (manure in Phase 2).

---

## Step 0 — Feasibility gate (DONE 2026-07-08)

Measured via `scripts/step0_timing.py` (default freestall scenario, 7 sim-years, in-process,
`workers=1`): **~3.8 s per simulated year, ~26 s per full run** — the "seconds" band. Consequences:
online RL via O(T) pause-and-resume is tractable (~120 transitions per 26 s run at monthly cadence →
~1e6 transitions ≈ ~60 h serial ≈ ~7–8 h across 8 subprocesses), while replay's O(T²) (~60× the
sim-work at T≈120) is fatal as a training path. Offline / growing-batch RL (IQL/CQL on logged
*simulator* trajectories — not a substitute for the simulator) is the fallback lever if online
wall-clock disappoints.

## Stepping strategy — two axes

RuFaS is a batch, run-to-completion simulator, so "step one period" is not native. The problem
decomposes into two independent axes; verifications #7/#10 guard every mode against dynamics drift.

**Axis A — the pause.** Primary: the **threaded pause hook** (zero RuFaS edits). Run
`TaskManager.start` on a worker thread; monkey-patch the daily-step method
(`_execute_full_farm_daily_simulation`, `simulation_engine.py`) so that at each decision boundary the
worker `put`s the live engine on a handoff queue and blocks until the harness sends the action — a
strict handoff (only one side ever runs), so the GIL and RuFaS's globals are fine. Carryover state
needs no serialization: it lives on the paused engine object graph (`self.field_manager`,
`self.herd_manager`, `self.time`, …). The boundary is a **cadence predicate** (year start / ~30 days
elapsed), so yearly and monthly share one mechanism. `reset()` teardown unwinds the worker via a
sentinel raising inside the hook; the subprocess-per-episode rule makes teardown forgiving. Fallback
(behind the same `Stepper` interface): the flag-gated generator refactor of
`_run_simulation_main_loop` (`simulation_engine.py:314`) — a small, reviewable, off-by-default source
patch, used only if the hook proves unworkable.

**Axis B — applying the action at the pause (the part that needed proving).** Mid-run pool injection
is a no-op (config is cached at construction — see Feasibility), so each lever is applied by
**re-invoking RuFaS's own construction-time setup function** against the live objects, via a
declarative **applier registry** `{lever: (config fragment, setup fn, assignment targets)}`:
- *Rations* — `RationManager.set_user_defined_rations` + `set_user_defined_ration_tolerance` from a
  modified `feed.rations` fragment. **PROVEN equivalent to construction injection**
  (`scripts/spike_axisb_rations.py`: 0 physical diffs). The only mid-year lever → what monthly
  cadence actually controls between year boundaries.
- *Crop / fertilizer / manure* — `FieldManager._setup_crop_events` / `_setup_fertilizer_events`
  (returns mixes **and** events — reassign both) / `_setup_manure_events`, reassigned onto each live
  `Field` at a **year boundary**. Replace only future-dated events (past events' effects are already
  in the soil state). **Spike pending — the next go/no-go.** If it fails, field-op levers fall back
  to construction-time injection (settable per episode, not per period) — a design constraint, not a
  dead end.

**Replay / prefix-simulation — demoted to correctness oracle.** Re-run RuFaS from period 0 with all
actions so far injected at construction (`input_patch` — the one proven-by-design injection path);
pin per-episode stochastic draws at `reset()` so prefixes are well-defined. O(T²) makes it unusable
for training at T≈120, but it is the ground truth every stepped mode must reproduce exactly (V7/V10),
and it needs zero RuFaS edits.

**Last resort — learned surrogate.** Unchanged: if online stepping somehow fails entirely, train a
fast neural emulator on logged trajectories, do RL on it, validate the final policy on real RuFaS.

---

## Action injection at construction (episode start; also the replay oracle's path)

An action is a nested-dict patch deep-merged into `InputManager.pool` before the simulation modules
are constructed:

1. Everything the simulation reads lives in `InputManager.pool` (loaded from JSON at run start).
   Modules call `get_data("animal.herd_information.cow_num")` — a dot-path lookup returning a
   `deepcopy` (`input_manager.py:1021`). Whatever is in `pool` at construction *is* the scenario.
2. The override channel is `input_patch`, deep-merged at `task_manager.py:1015`.
   `flatten_keys_to_nested_structure` (`util.py:65`) maps a flat RL action vector into nested form.
   Sensitivity analysis already drives this path (`task_manager.py:499-503`).
3. Because `input_patch` is `None` for single runs (`task_manager.py:373`), the harness sets it via
   monkey-patch, as `pool_store.py:65-78` does — set `args["input_patch"]` to the decoded action just
   before the engine builds. (On-disk alternative: materialize the action to JSON blobs, as
   `scenario_store._materialize()` does.)
4. The deep-merge runs before `field_manager._setup_*` and `HerdManager` construction freeze every
   schedule and ration, so a single up-front patch controls every lever for the whole horizon. It is
   **construction-time only**: nothing re-reads the pool mid-run — rations do NOT re-read on
   reformulation (verified; `formulate_rations` consumes cached `RationManager` class state).
   Mid-episode decisions therefore go through the Axis-B appliers (see Stepping strategy), not the
   pool.

---

## Architecture

Python package **`rufas_rl/`** (own repo, depending on `../RuFaS` via
`rufas-web/api/rufas_bootstrap.py`'s sys.path shim). Public class `RufasEnv(gymnasium.Env)`.
Components are pluggable strategy objects — **observer / implementer / rewarder / constrainer** — on
Gymnasium (5-tuple `terminated`/`truncated`).

### Phase 1 — Single-step env + baselines

- **`reset(seed)`**: flush singletons, load a base scenario pool, set a nonzero seed. (Scenario
  sampler is an optional hook here; a fixed base scenario suffices.)
- **`step(action)`**: decode action → build `input_patch` → run the full horizon via
  `TaskManager.start` → read `variables_pool` → compute reward + constraints → `terminated=True`.
- **Action encoding** (heterogeneous → flatten to `Box` for SB3):
  - *Feed ratios*: per-group simplex (`calf/growing/close_up/lac_cow`) via softmax → sums to 100
    (vs. `ration_manager.py` tolerance). Writes `feed.rations[*].feeds[*].ration_percentage`.
  - *Crop rotation*: categorical per year (argmax over
    `crop_configurations/default_crop_configs.json`). Also the N-fixation lever — legume fixation
    follows from `is_nitrogen_fixer=true`. Writes `crop_schedules`.
  - *Fertilizer*: continuous N/P amounts on fixed timing templates. Writes `fertilizer_schedule`.
  - `needs_rerun` flag — skip re-simulation when an action doesn't alter the current period.
- **Reward** (`reward_fn(pool) -> float | vector`): economics from EEE, enteric methane from
  `digestive_system/enteric_methane_calculator`, field/manure emissions from `EEE/emissions.py`,
  soil-N leaching from `soil/nitrogen_cycling/`. Presets: `profit`, `profit_minus_ghg`,
  `profit_n_constrained`, `weighted_vector`. Average N runs.
- **Constraints** (optional): `compute_constraint(pool)` returns per-step constraint values in `info`
  for constrained/safe RL. Otherwise fold caps into a `weighted_vector` reward.
- **Baselines:** CMA-ES / Optuna static-plan + greedy.
- **Output capture:** default `pool_store` monkey-patch; native output files for max purity.

### Phase 2 — Sequential (event-stepped) MDP

The agent observes realized carryover state each period and chooses the next period's decisions.
Implement the stepping strategy: the threaded pause hook + applier registry at **monthly** cadence
(yearly via the same predicate for bring-up/ablation), gated by the Axis-B field-op spike and the
replay-oracle equivalence tests (V7/V10). Per-boundary action masking: rations settable at every
boundary; crop/fertilizer/manure only at year boundaries. Add **manure application** to the action
space (`manure_schedule`: N/P amounts + template timing + placement fractions). Train the policy
**online** (PPO / SAC via Stable-Baselines3) and compare against the Phase-1 baselines. Drive the
`TaskManager.start` path — not a bare `SimulationEngine`.

### Phase 3 — Robustness, adaptation study, parallelism

- **Scenario sampler:** pluggable object that samples weather/price/init realizations at each
  `reset()` (via `input_patch` / swapping the `weather` blob). Wired from Phase 1.
- **Averaging:** N-run averaging per evaluation (in the rewarder).
- **Parallelism:** one episode per fresh subprocess is already the serial rule (in-process reuse
  degrades ~5×/run; the `InputManager`/`OutputManager` singletons are unsafe to share — RuFaS uses
  `maxtasksperchild=1`). Scale-out = many episode subprocesses at once, via Gymnasium
  `AsyncVectorEnv` / SB3 `SubprocVecEnv`.
- Optional: surface training runs / rollouts through the `rufas-web` FastAPI + React UI.

---

## Critical files

**New (`rufas_rl/`):**
- `env.py` — `RufasEnv(gymnasium.Env)` (reset/step/spaces; online-stepped mode with cadence knob;
  replay mode kept as the verification oracle).
- `implementers.py` — decode action → `input_patch`; `needs_rerun` flag.
- `observers.py` — extract `s_t` (soil/herd/inventory carryover) from `variables_pool`.
- `rewarders.py` — reward presets + N-run averaging.
- `constrainers.py` — per-step constraint values for constrained/safe RL.
- `scenario.py` — weather/price/init scenario sampler (per-reset draws).
- `harness.py` — subprocess-per-episode runner (singleton flush, `TaskManager.start` + chdir, pool
  capture), construction-time `input_patch` injection, the threaded-pause `Stepper` (worker thread +
  handoff queues + cadence predicate), and the **applier registry** (live-mutation setup-fn calls per
  lever). Adapts `pool_store.py`, `runner.py`.
- `experiments/` — CMA-ES static + greedy baselines, PPO policy, comparison eval.
- `scripts/` — feasibility spikes. Done: `step0_timing.py` (V1) and `spike_axisb_rations.py` (ration
  applier ≡ construction injection). Next: the field-op applier spike and the Axis-A pause spike.

**RuFaS source (fallback only — the threaded hook needs zero edits):**
- `simulation_engine.py` — the flag-gated generator refactor of `_run_simulation_main_loop` (`:314`),
  used only if the threaded pause hook proves unworkable. Off by default = byte-identical to released
  RuFaS; kept as a small, reviewable diff. No other RuFaS file is ever edited.

**Reused (unchanged):** `rufas-web/api/pool_store.py` (capture hook + patch merge), `runner.py`
(`TaskManager.start` call + chdir), `editable_inputs.py` (override keys + bounds),
`RuFaS/input/metadata/properties/default.json` (schema bounds), `../cyclesgym` (design precedent).

---

## Verification

1. **Step 0 timing** — DONE: ~3.8 s/sim-year, ~26 s/run (`scripts/step0_timing.py`); corollary
   finding: in-process reuse degrades ~5×/run → one episode per fresh subprocess.
2. **Dynamics unchanged:** the threaded hook, appliers, and replay oracle keep `git status` in
   `../RuFaS` clean; if the fallback source patch is used, the diff is confined to the stepping
   refactor and is byte-identical with the flag off.
3. **Reset hygiene:** same seed + action → near-identical reward across two `reset()/step()` calls.
4. **Action plumbing:** a known patch (bump fertilizer N) measurably shifts the corresponding
   `variables_pool` outputs.
5. **Reward sanity:** an obviously-good vs. obviously-bad action ranks correctly.
6. **Sequential dynamics:** a past action changes a later period's state/reward (legume in year 1
   raises year-2 corn yield / lowers year-2 fertilizer).
7. **Applier equivalence (per lever):** a run whose lever is live-mutated mid-run reproduces a
   construction-injected monolithic run's physical `variables_pool` exactly. **Rations: PASS**
   (`scripts/spike_axisb_rations.py`, 0/2,701 physical diffs). Field ops: pending — the go/no-go for
   per-period field control.
8. **Headline result:** the learned policy beats greedy and static-CMA-ES baselines, stable across
   seeds.
9. **Subprocess isolation:** N parallel subprocess envs equal N serial runs.
10. **Stepped-run correctness:** an online-stepped episode (pause → mutate → resume) reproduces the
    monolithic/replay-oracle run's `variables_pool` exactly given the same decisions; if the fallback
    source patch is used, its flag-off path stays byte-identical.
