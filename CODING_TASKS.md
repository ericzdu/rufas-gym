# Coding Tasks — RuFaS RL Environment

An assignment-style build guide. Each task lists **what to code**, **requirements**, and **done
when** (acceptance criteria, cross-referenced to the plan's verifications V1–V10). Do the tasks in
order; the critical path is T0 → T1 → T2/T3/T4 → T5 → T7 → T9. Companion docs: `ARCHITECTURE.md`
(concepts), `RL_ENVIRONMENT_PLAN.md` (full spec + line anchors), `IMPLEMENTATION_ORDER.md` (sequence).

**Ground rules (apply to every task):**
- Never edit RuFaS physics. The primary stepper (T7's threaded pause hook) edits zero RuFaS files;
  the only permitted source edit is the T8 fallback patch, used only if the hook proves unworkable.
- **Mid-run pool injection is a no-op** (config is cached at construction — verified). `input_patch`
  sets levers at episode start; mid-episode changes go through the Axis-B appliers (re-invoke RuFaS's
  own setup functions on live objects), each gated by an equivalence spike.
- **One episode per fresh subprocess** — in-process reuse degrades ~5×/run even after flushing both
  singleton pools.
- All RuFaS access goes through `TaskManager.start` — never construct a bare `SimulationEngine`.
- Re-verify every cited line number against `../RuFaS` before relying on it; they drift.
- Every task that produces behavior ships with a test under `rufas_rl/tests/`.

---

## T0 — Project setup & feasibility gate

**What to code:** the repo skeleton and a timing smoke test.

**Requirements:**
1. Create repo `rufas_rl/` with a `uv`-managed `pyproject.toml`. Dependencies: `gymnasium`,
   `stable-baselines3`, `cma`, `optuna`, `numpy`, and dev deps `pytest`, `tensorboard`.
2. Import RuFaS via `rufas-web/api/rufas_bootstrap.py`'s sys.path shim (put `../RuFaS` on `sys.path`).
   Pin RuFaS's Python version and requirements; our env must be a superset.
3. Write `scripts/step0_timing.py`: run one full multi-year `TaskManager.start` single-run on a base
   scenario, in-process, and print **per-simulated-year wall-clock**.

**Done when (V1):** ✅ **DONE (2026-07-08).** `scripts/step0_timing.py` records **~3.8 s/sim-year
(~26 s per 7-year run)** — see README. Consequence: online O(T) stepping is the training path; replay
is the oracle only. Bonus finding: a second in-process run is ~5× slower (state leak) →
subprocess-per-episode (see T1). Still open from this task: the `uv` repo skeleton
(`pyproject.toml` + deps).

---

## T1 — `harness.py` (foundation; everything depends on it)

**What to code:** a thin, synchronous wrapper that runs RuFaS once with an injected action and returns
its raw output pool.

**Requirements:**
1. `reset_singletons()` — call `InputManager.flush_pool` (`input_manager.py:1317`) and
   `OutputManager.flush_pools` (`output_manager.py:2202`) so no state leaks between runs.
2. `load_base_pool(scenario_path) -> dict` — load the base scenario into the `InputManager.pool`.
3. `run(input_patch: dict) -> dict` — inject `input_patch` and run the full horizon:
   - Monkey-patch the handler so `args["input_patch"]` is set to `input_patch` just before the engine
     builds (mirror `pool_store.py:65-78`); do **not** rely on the public task JSON, since
     `input_patch` is hard-set to `None` for single runs (`task_manager.py:373`).
   - Invoke `TaskManager.start` synchronously with `workers == 1` (in-process; see
     `task_manager.py:176`, `~:528`). Handle `chdir` as `runner.py` does.
   - Capture and return `variables_pool` via `output_manager.py:2181` `_get_flat_variables_pool`.
4. **Run each episode in a fresh subprocess.** Flushing both singleton pools is NOT sufficient — a
   second in-process run is ~5× slower with identical outputs (verified). Repeated `run()` calls from
   the parent must be subprocess-isolated; the flush still runs inside each child for hygiene.

**Done when (V3):** ✅ **DONE.** Built as `harness.py` (`EpisodeProcess`) + `worker.py`, using a
`spawn` context so no RuFaS module state is inherited. `reset(seed=123)` twice returns identical
initial observations (`test_env_slow.py::test_same_seed_reproduces_the_same_start`), and
`gymnasium`'s own `check_reset_seed_determinism` passes. Note the shape differs from the original
sketch: because stepping needs a *live* paused thread, the child hosts the whole `Episode` and the
parent drives it over a pipe, rather than the child running one shot and returning a pool.

---

## T2 — `implementers.py` (action → `input_patch`)

**What to code:** encode an RL action vector into a nested `input_patch` dict.

**Requirements:**
1. Define the action space as a flat `gymnasium.spaces.Box` (SB3-friendly). Implement
   `decode(action: np.ndarray) -> dict` producing per-lever config fragments, assembled into an
   `input_patch` (episode start / replay oracle) or handed to the T7 appliers (mid-episode) — same
   decode, two consumers.
2. **Feed ratios:** per animal group (`calf`, `growing`, `close_up`, `lac_cow`), map a raw sub-vector
   through **softmax** to a simplex summing to 100. Write to
   `feed.rations[*].feeds[*].ration_percentage`. Validate the sum against `ration_manager.py`'s
   tolerance.
3. **Crop rotation:** categorical per year via `argmax` over the options in
   `crop_configurations/default_crop_configs.json`. Write to `crop_schedules`. Do **not** add a
   separate N-fixation lever — legume fixation follows from `is_nitrogen_fixer=true` on the chosen
   crop.
4. **Fertilizer:** continuous N/P amounts on fixed timing templates. Write to `fertilizer_schedule`.
5. Use `flatten_keys_to_nested_structure` (`util.py:65`) if emitting flat dot-path keys; ensure the
   result deep-merges correctly (`deep_merge`, `util.py:340`, merges lists element-wise).
6. Return a `needs_rerun: bool` alongside the patch — `False` when an action does not change the
   current period (lets the env skip re-simulation).
7. Clamp all continuous values to the bounds in `editable_inputs.py` / `default.json`.

**Done when (V4):** ✅ **DONE for the ration lever** (`implementers.py`). Per-group softmax onto a
simplex summing to 100, so *every* point in the action space is a valid ration — no clipping or
masking. The space is the symmetric unit box SB3 expects, with logits scaled internally (×5) so a
near-one-hot ration is still reachable. A lopsided ration measurably changes the trajectory
(`test_env_slow.py::test_action_changes_the_trajectory`). Crop/fertilizer/manure encodings are
**not** built — they wait on the field-op applier spike, and `EnvConfig` rejects them rather than
accepting them silently.

---

## T3 — `observers.py` (RuFaS output → state `s_t`)

**What to code:** extract the agent-visible state vector from `variables_pool`.

**Requirements:**
1. `observe(pool: dict, period: int) -> np.ndarray` returning a fixed-length vector for the given
   period.
2. Include, per field: soil N/P/C pools and moisture; standing crop & residue. Herd: counts by age &
   lactation stage, body condition. Inventory: feed/manure storage. Calendar/season.
3. Define a matching `gymnasium.spaces.Box` observation space with sane bounds; normalize features to
   comparable scales.
4. Be robust to missing/again-null keys (some pools may not populate every field early in bring-up).

**Done when:** ✅ **DONE** (`observers.py`), but built against **live engine objects, not the
`variables_pool`** — a deliberate departure from this task's original wording. Measured: the pool
does accumulate mid-run (134 keys at the first pause → 1,740 by the 14th), but its soil variables
come from *annual* reporters, so at a monthly pause they are up to eleven months stale and blind to
the agent's own recent actions. That would quietly break the Markov property. The live object graph
(`engine.field_manager.fields[i].soil.data`, `crop.data`, `growth_constraints`,
`herd_manager.herd_statistics`) is the true current state; reading it is pure inspection. 54 features
on the default scenario: calendar (3) + per-field soil/crop/stress (20 × n_fields) + herd (11),
normalized and clipped into the declared Box.

---

## T4 — `rewarders.py` (RuFaS output → reward)

**What to code:** pluggable reward functions plus multi-run averaging.

**Requirements:**
1. `reward_fn(pool: dict) -> float | np.ndarray`. Implement presets:
   - `profit` — economics from the EEE module.
   - `profit_minus_ghg` — profit minus enteric methane
     (`digestive_system/enteric_methane_calculator`) and field/manure emissions (`EEE/emissions.py`).
   - `profit_n_constrained` — profit with an N-leaching penalty (`soil/nitrogen_cycling/`).
   - `weighted_vector` — return the component vector for multi-objective weighting.
2. `averaged_reward(patch, n_runs)` — call the harness `n_runs` times and average the reward, to damp
   RuFaS's mild stochasticity.
3. Reward presets are selected by config, not hardcoded into the env.

**Done when (V5):** ⚠️ **PARTIAL.** `rewarders.py` ships the pluggable interface and one preset,
`milk_minus_nitrogen` — an explicit **bring-up proxy, not the paper's objective**. It prices milk
against nitrate runoff and ignores feed cost, herd economics and GHG entirely. The EEE-based presets
remain unbuilt, and there is a structural reason to flag: **EEE economics and emissions are reported
annually**, so a faithful `profit` reward is *sparse* at monthly cadence. Resolving that (annual
reward with monthly actions, or reward shaping) is a real open design question, not just coding.
Multi-run averaging is also not implemented.

---

## T5 — `env.py` Phase 1 (single-step `RufasEnv`)

**What to code:** the Gymnasium env for one-decision episodes.

**Requirements:**
1. `class RufasEnv(gymnasium.Env)` with `action_space` (from T2) and `observation_space` (from T3).
2. `reset(seed)` — `harness.reset_singletons()`, load the base scenario pool, set a nonzero seed,
   return the initial observation + `info`. Accept an optional scenario-sampler hook but default to a
   fixed base scenario.
3. `step(action)` — `implementer.decode` → `harness.run` → `observer.observe` +
   `rewarder.reward_fn` (+ optional constrainer) → return
   `(obs, reward, terminated=True, truncated=False, info)`. Surface constraint values and raw
   economics/emissions in `info`.
4. Pass the standard `gymnasium.utils.env_checker`.

**Done when (V3, V5):** ✅ **DONE** (`env.py`) — though built directly as the *sequential* env (T7)
rather than the single-step Phase-1 env, since the pause mechanism was already proven and a
one-decision env would have been throwaway. `gymnasium.utils.env_checker` passes; V3 holds. One
gotcha worth recording: `env.spec` is reserved by Gymnasium for its `EnvSpec`, so the scenario spec
is exposed as `env.scenario_spec` — the checker fails loudly otherwise.

---

## T6 — `experiments/` baselines

**What to code:** the two non-RL baselines the paper compares against.

**Requirements:**
1. **Static full-horizon optimizer:** encode the whole plan as one parameter vector, optimize with
   **CMA-ES** (or Optuna) against episode return on `RufasEnv`.
2. **Greedy/myopic optimizer:** optimize each period independently, ignoring carryover.
3. Log returns and wall-clock; fix seeds for reproducibility.

**Done when:** both baselines run end-to-end on the Phase-1 env and produce a repeatable return the
learned policy will later be measured against.

---

## T7 — `env.py` Phase 2 (sequential MDP, online stepping)

**What to code:** extend `RufasEnv` to multi-step episodes via **online pause-and-resume** — the
threaded pause hook (Axis A) plus the applier registry (Axis B).

**Requirements:**
1. **Prerequisite spikes** (small scripts under `scripts/`; PASS required before wiring in):
   - *Rations applier* — ✅ **DONE** (`scripts/spike_axisb_rations.py`): live-mutating
     `RationManager.set_user_defined_rations`/`set_user_defined_ration_tolerance` mid-run ==
     construction injection (0 physical diffs of 2,701 keys).
   - *Field-op appliers* — same 3-way equivalence design for `FieldManager._setup_crop_events` /
     `_setup_fertilizer_events` (reassign mixes **and** events) / `_setup_manure_events`, mutated
     onto live `Field`s at a year boundary; replace only future-dated events. Go/no-go for
     per-period field control (fallback: field levers set per-episode at construction).
   - *Axis-A pause* — ✅ **DONE** (`scripts/spike_axisa_pause.py`): the threaded pause hook (worker
     thread + monkey-patched `_execute_full_farm_daily_simulation` blocking on a hand-off queue at a
     monthly cadence predicate) pauses the 7-yr `example_freestall` run 84× and resumes cleanly (no
     deadlock); its `variables_pool` is byte-identical to an unhooked run (0/2702 physical keys
     differ); STOP sentinel tears down the worker cleanly. Ready to wrap as `ThreadedPauseStepper`.
2. **`Stepper` interface** in `harness.py`; first implementation `ThreadedPauseStepper`: run
   `TaskManager.start` on a worker thread; monkey-patch `_execute_full_farm_daily_simulation` to
   block on a handoff queue at decision boundaries; `step(action)` = release the worker with the
   action → block until the next boundary. Teardown on `reset()` via a sentinel raising inside the
   hook. The whole episode (thread included) lives inside the T1 subprocess.
3. **Cadence is a boundary predicate**, config-selected: `monthly` (~30 days, aligned with
   `formulation_interval`; the primary target) and `yearly` (coarse case) share one mechanism.
   Per-boundary action masking: rations settable at every boundary; crop/fertilizer/manure only at
   year boundaries.
4. **Applier registry** `{lever: (config fragment, setup fn, assignment targets)}` — apply the T2
   fragments at each pause by re-invoking RuFaS's own setup functions on the live engine.
5. **Determinism:** pin every per-episode stochastic draw at `reset()` (same seed/weather) so stepped
   runs and oracle runs are comparable and the MDP is well-defined.
6. `terminated=True` only at the horizon end (`T` periods); `truncated` per Gymnasium convention.
   Honor `needs_rerun=False` where applicable.
7. Add **manure application** to the action space (`manure_schedule`: N/P amounts, template timing,
   placement fractions) via the T2 implementer.
8. Keep a `mode="replay"` path (accumulate actions; re-run from period 0 via one construction-time
   `input_patch`) — **not for training** (O(T²) at T≈120); it is the **oracle** for V7/V10 and the
   reference implementation for tests.

**Done when (V6, V7, V10):** ⚠️ **PARTIAL — the stepper ships; the field levers and the oracle do
not.**

Shipped (`stepper.py`, `episode.py`, `env.py`):
- `ThreadedPauseStepper` wraps the proven hook, now patching **all four** dispatchable daily methods
  rather than just `_execute_full_farm_daily_simulation`. That closes a real trap: hooking only the
  full_farm method meant a `field_only` scenario ran to completion, looked entirely normal, and
  **silently never paused**. `scripts/verify_env.py pauses` now asserts a nonzero pause count per
  scenario — full_farm 84x, field_only (Kimberly) 96x.
- Dynamics preservation re-proven for the shipped code, not just the spike:
  **0 of 2843 physical keys differ** between a stepped and an unhooked run
  (`scripts/verify_env.py equivalence`).
- Cadence is config-selected (`monthly` / `yearly` / `daily`) through one predicate.
- Ration applier wired; `EnvConfig` **rejects** the unproven levers instead of accepting them.

Still open:
- **V7 field-op appliers** — crop/fertilizer/manure still have no equivalence spike, so field levers
  are construction-time only. This is the blocker for the thesis levers.
- **V6** — untestable as a cross-*year* claim until those levers exist. Rations do affect later
  periods, but the legume→next-year-corn story needs the field appliers.
- **V10 replay oracle** — `mode="replay"` is not implemented, so there is no independent
  reconstruction of a stepped episode. The equivalence check above is the weaker (but real) property:
  pausing doesn't perturb a run. Per-boundary action masking is also not implemented (moot while
  rations are the only lever, since they are settable at every boundary).

---

## T8 — Fallback source patch (CONDITIONAL — only if T7's threaded hook proves unworkable)

**What to code:** the minimal RuFaS source edit that makes the simulation loop resumable — pursued
only if the zero-edit threaded pause hook turns out unmanageable in practice (messy teardown,
deadlocks, un-debuggable state).

**Requirements:**
1. In `simulation_engine.py`, refactor `_run_simulation_main_loop` (`:314`) so the loop yields
   control at decision boundaries — a generator/step interface, or hoist the engine so a driver calls
   "advance one period." **Monthly cadence needs the yield inside the daily loop, not just per year**
   — budget accordingly. Gate it behind a flag that defaults **off**; flag-off must run the original
   code path unchanged.
2. Keep the diff small, confined, and documented as a reviewable patch.
3. Reuse T7's applier registry unchanged at each yield — Axis B is identical regardless of the pause
   mechanism. (NOTE: rations do NOT re-read the pool on reformulation — verified; every lever goes
   through its applier.)
4. Add `mode="stepping_patch"` to `env.py` selecting this path (same `Stepper` interface as T7).

**Done when (V2, V10):** with the flag off, RuFaS is byte-identical to released (V2); a stepped run
reproduces the replay-oracle run's `variables_pool` exactly (V10).

---

## T9 — Training & comparison (the headline)

**What to code:** train the RL policy and compare against baselines.

**Requirements:**
1. Train **PPO** and/or **SAC** (Stable-Baselines3) on the sequential `RufasEnv`.
2. Evaluate the learned policy and both T6 baselines on the same held-out scenarios/seeds with N-run
   averaging.
3. Report all three objectives (`profit`, `profit_minus_ghg`, N-leaching).
4. Produce comparison plots + a results table; log to TensorBoard.

**Done when (V8):** the learned policy beats the greedy and static-CMA-ES baselines, stable across
seeds.

---

## T10 — Robustness, adaptation study, parallelism

**What to code:** `scenario.py` sampling, averaging, and vectorized rollouts.

**Requirements:**
1. `scenario.py` — a pluggable sampler drawing weather/price/init realizations at each `reset()` (via
   `input_patch` / swapping the `weather` blob). Wire it into `reset()`.
2. Confirm N-run averaging (T4) is applied in evaluation.
3. **Parallelism:** subprocess-per-episode is already the T1 rule; scale-out = many episode
   subprocesses at once via Gymnasium `AsyncVectorEnv` / SB3 `SubprocVecEnv`.
4. Optional: a "value of adaptation" analysis (adaptive policy vs. best static plan under sampled
   scenarios).
5. Optional: surface runs/rollouts through the `rufas-web` FastAPI + React UI.

**Done when (V9):** N parallel subprocess envs produce the same results as N serial runs.

---

## Verification checklist (V1–V10)

| V | Check | Task |
|---|---|---|
| V1 | Step-0 per-year timing recorded — ✅ DONE (~3.8 s/yr, ~26 s/run) | T0 |
| V2 | Dynamics unchanged (git clean; fallback flag-off byte-identical) | T1, T8 |
| V3 | Reset hygiene across fresh subprocesses (same seed+action → same reward) — ✅ DONE | T1, T5 |
| V4 | Action plumbing shifts the right outputs — ✅ DONE (rations) | T2 |
| V5 | Reward ranks good vs. bad correctly — ⚠️ proxy reward only | T4, T5 |
| V6 | A past action changes a later period — ⚠️ blocked on field appliers | T7 |
| V7 | Applier ≡ construction injection, per lever (rations ✅; field ops pending) | T7 |
| V2b | Stepped run ≡ unhooked run — ✅ DONE, 0/2843 keys differ | T7 |
| V8 | Learned policy beats both baselines | T9 |
| V9 | Parallel subprocess envs equal serial runs | T10 |
| V10 | Online-stepped run reproduces the replay oracle | T7, T8 |
