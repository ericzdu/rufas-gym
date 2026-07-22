# Implementation Order — RuFaS RL Environment

Companion to `RL_ENVIRONMENT_PLAN.md` (the *what/why*, which holds the authoritative line-anchor
catalog). This doc is the *order* and *what-touches-what*: **RuFaS engine touchpoints**, **`rufas_rl/`
files**, and **dependencies**, with a milestone sequence mapped to the plan's verifications.

**Editing scope:** RuFaS dynamics stay fixed. The primary online stepper (threaded pause hook,
touchpoint 7) edits zero RuFaS files (`git status` clean); the only permitted source edit is its
flag-gated generator fallback, used only if the hook proves unworkable — off by default it is
byte-identical to released RuFaS. Action injection stays a monkey-patch at construction; mid-episode
actions are applied by re-invoking RuFaS's own setup functions on live objects (touchpoint 7a).

---

## 1. Dependencies & environment (first — blocks everything)

Use **`uv`**. New repo `rufas_rl/` with its own `pyproject.toml`.

- **RuFaS** — not pip-installed; imported via `rufas-web/api/rufas_bootstrap.py`'s sys.path shim
  (`../RuFaS` on `sys.path`). Pin RuFaS's Python version + `requirements`; our env is a superset.
- **`gymnasium`** — env API (5-tuple).
- **`stable-baselines3`** — PPO / SAC (Phase 2). Pulls `torch`.
- **`cma`** and/or **`optuna`** — baselines (Phase 1).
- **`numpy`** — action/obs vectors (pin compatibly with RuFaS).
- **Optional:** `tensorboard`, `pytest`.

**Gate:** ✅ DONE — `scripts/step0_timing.py` imports RuFaS through the shim, completes
`TaskManager.start` runs, and records **~3.8 s/sim-year (~26 s per 7-year run)**. Result: online O(T)
stepping is the training path (replay = oracle only), and the ~5× in-process degradation makes
subprocess-per-episode a Phase-1 rule. Remaining from this step: the `uv` repo skeleton.

---

## 2. RuFaS engine touchpoints — sequenced

Sequencing view of the plan's Feasibility catalog (full line anchors there; re-verify before use).
Touchpoints 1–6 are hooks/reads (no source edit); 7 is the primary threaded stepper (also no source
edit; its generator fallback is the one conditional source patch); 7a is the applier registry.
Wrapping and reads live in `rufas_rl/harness.py`, adapting `rufas-web/api/pool_store.py`.

| # | Touchpoint | Location | When |
|---|------|----------|------|
| 1 | Singleton flush | `input_manager.py:1317` `flush_pool`, `output_manager.py:2202` `flush_pools` | `reset()` hygiene. Phase 1. |
| 2 | Base scenario in `pool` | `InputManager.pool` (loaded from JSON at run start) | Load base scenario at `reset()`. Phase 1. |
| 3 | Action injection *(construction-time ONLY)* | deep-merge `input_patch` into `pool` at `task_manager.py:1015`; `input_patch` is `None` for single runs at `task_manager.py:373` | Set `args["input_patch"]` before the engine builds, as `pool_store.py:65-78` does. Works only before module construction — mid-run pool edits are a no-op (verified). Phase 1. |
| 4 | Deep-merge semantics | `deep_merge` `util.py:340`; `flatten_keys_to_nested_structure` `util.py:65` | Maps a flat action vector onto nested schedule/ration blobs. Read-only. Phase 1. |
| 5 | In-process single-worker path | `task_manager.py:176` (pool→`None` when `workers==1`), `~:528` (`list(map(...))`) | Confirms in-process run so patch + capture work. Phase 1. |
| 6 | Output capture | `output_manager.py:2181` `_get_flat_variables_pool` | Read `variables_pool` for `s_{t+1}` + reward. Phase 1. |
| 7 | Threaded pause hook *(primary stepper; zero source edits)* | monkey-patch `_execute_full_farm_daily_simulation` (`simulation_engine.py`); run `TaskManager.start` on a worker thread | Worker blocks on a handoff queue at each decision boundary (cadence predicate: monthly primary, yearly coarse); harness reads state, applies the action (7a), releases. Fallback = flag-gated generator refactor of `_run_simulation_main_loop` (`:314`) — the one permitted source edit. Phase 2. |
| 7a | Applier registry *(live mutation via RuFaS's own setup fns)* | rations: `RationManager.set_user_defined_rations`/`set_user_defined_ration_tolerance`; field ops: `FieldManager._setup_crop_events`/`_setup_fertilizer_events` (reassign mixes AND events)/`_setup_manure_events` onto each live `Field` (`field_manager.py:150-200`) | Mid-run pool injection is a no-op — apply each lever by re-invoking its construction setup fn. Rations spike ✅ PASS (`spike_axisb_rations.py`, 0 physical diffs). Field-op spike = next go/no-go (replace future-dated events only). Phase 2. |

Setup context (read-only): `get_data()` runs only in `_setup_simulation_modules`
(`simulation_engine.py:207-278`), not the daily loop — carryover state lives in the engine object
graph. That is why the pause needs no serialization (the paused thread's engine IS the state) and why
mid-run pool edits change nothing.

---

## 3. `rufas_rl/` build order

Bottom-up: harness → implementer/observer/rewarder → env → baselines → trainer.

1. **`harness.py`** — subprocess-per-episode runner (singleton flush, `TaskManager.start` + `chdir`,
   pool capture), construction-time action injection; later, the `Stepper` interface
   (`ThreadedPauseStepper`: worker thread + handoff queues + cadence predicate) and the applier
   registry. Adapts `pool_store.py`, `runner.py`.
2. **`implementers.py`** — decode action → per-lever config fragments (feed simplex, crop categorical,
   fertilizer continuous), consumed as a construction `input_patch` or by the appliers mid-episode;
   `needs_rerun` flag. Validate encodings against `ration_manager.py` tolerance and
   `editable_inputs.py` bounds.
3. **`observers.py`** — extract `s_t` (soil N/P/C + moisture, herd structure, storage, calendar) from
   `variables_pool`.
4. **`rewarders.py`** — presets (`profit`, `profit_minus_ghg`, `profit_n_constrained`,
   `weighted_vector`) + N-run averaging. Sources: EEE economics, enteric methane, `EEE/emissions.py`,
   `soil/nitrogen_cycling/`.
5. **`env.py`** — `RufasEnv(gymnasium.Env)`: `reset`/`step`/spaces. Phase 1 = single-step; Phase 2
   adds online sequential stepping (threaded hook + appliers, cadence knob), with replay mode kept as
   the verification oracle.
6. **`constrainers.py`** (optional) — per-step constraint values in `info`.
7. **`scenario.py`** — weather/price/init sampler at each `reset()`. Hook wired in Phase 1; real
   sampling in Phase 3.
8. **`experiments/`** — CMA-ES/Optuna static + greedy baselines (Phase 1), PPO/SAC policy + comparison
   eval (Phase 2).

---

## 4. Milestone sequence

- **M0 — Setup gate.** ✅ Step-0 timing recorded (~3.8 s/sim-year); RuFaS imports via shim; runs
  complete. Remaining: the `uv` repo skeleton. *(Verif. 1, 2)*
- **M1 — Plumbing spike.** `harness.py` + `implementers.py`: inject a known patch, confirm it shifts
  the right `variables_pool` outputs. *(Verif. 4)*
- **M2 — Phase-1 env.** `observers` + `rewarders` + single-step `env.py`. *(Verif. 3, 5)*
- **M3 — Baselines.** CMA-ES/Optuna static + greedy in `experiments/`.
- **M4 — Sequential env (online stepping).** Prerequisite spikes first (rations ✅ done; field-op
  appliers; Axis-A pause), then `ThreadedPauseStepper` + applier registry at monthly cadence (yearly
  knob); a past action changes a later period; an online-stepped episode reproduces the replay oracle
  exactly. *(Verif. 6, 7, 10)*
- **M4′ — Fallback source patch** (only if the threaded hook proves unworkable). Flag-gated
  resumable-loop refactor reusing the same appliers; flag-off byte-identical. *(Verif. 10)*
- **M5 — Train & compare.** PPO/SAC vs. greedy + static-CMA-ES. *(Verif. 8)*
- **M6 — Robustness & scale.** `scenario.py` sampling, N-run averaging, subprocess-per-env rollouts.
  *(Verif. 9)*

**Critical path:** M0 → M1 → M2 → M4 → M5. M3 runs parallel to M4. M6 is post-headline.
