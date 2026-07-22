# rufas-gym

A dynamics-preserving reinforcement-learning environment wrapping the validated
RuFaS whole-farm dairy simulator. See `ARCHITECTURE.md` (concepts),
`RL_ENVIRONMENT_PLAN.md` (spec), `IMPLEMENTATION_ORDER.md` (sequence), and
`CODING_TASKS.md` (build guide).

## The environment

```python
from rufas_rl import RufasEnv, EnvConfig

env = RufasEnv(EnvConfig(cadence="monthly"))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

One episode is one multi-year RuFaS simulation, paused at every decision boundary.
The agent sees the farm's real carryover state at each pause and sets the next
period's ration; the simulator runs its daily loop autonomously in between. On the
default freestall scenario that is **84 monthly decisions over 7 years**, with a
54-dimensional observation and a 23-dimensional action.

**RuFaS is not modified.** `git status` in `../RuFaS` is unchanged by anything here.

| module | role |
|---|---|
| `spec.py` | scenario shape read straight from its JSON — lets the spaces exist before any run |
| `stepper.py` | Axis A: the threaded pause hook |
| `appliers.py` | Axis B: applies an action by re-invoking RuFaS's own setup functions |
| `implementers.py` | action vector → ration percentages (per-group softmax simplex) |
| `observers.py` | paused engine → state vector |
| `rewarders.py` | interval flows → reward |
| `episode.py` | one episode, in-process (no IPC — the debuggable path) |
| `worker.py` / `harness.py` | one episode per fresh subprocess |
| `env.py` | `RufasEnv(gymnasium.Env)` |

### Install and verify

```
uv venv && uv pip install -e . && uv pip install pytest
uv run pytest rufas_rl/tests -m "not slow"   # 38 tests, ~0.2 s
uv run pytest rufas_rl/tests -m slow         # runs real simulations
uv run python scripts/verify_env.py          # the two load-bearing properties
```

### Experiment 1 — ration optimization

```
uv pip install cma
uv run python experiments/optimize_ration.py --years 7 --budget 48 --out results/ration_optimization_7yr.json
uv run python experiments/make_figures.py --results results/ration_optimization_7yr.json
uv run python experiments/make_report.py  --results results/ration_optimization_7yr.json
```

CMA-ES over the herd's ration, maximizing profit (milk revenue − RuFaS's *own* feed cost),
seeded at the farm's configured ration. Over a 7-year horizon it finds **+36% profit
($761,844)** versus the farm's ration — entirely by cutting feed cost **39%** at **0.0%**
milk change. The mechanism: RuFaS milk is largely insensitive to ration composition, so
profit-optimal feeding shifts weight off the $1.00/kg concentrates onto the $0.01/kg
forages. This is a **static, single-lever baseline** — the number a sequential RL policy
must later beat, not the RL result itself. `make_report.py` renders the figures + a
step-by-step derivation into a self-contained `results/report.html`.

### What is verified

| check | result |
|---|---|
| Stepped run vs unhooked run | **0 of 2843 physical keys differ** — pausing does not change the science |
| Every simulation type pauses | full_farm 84x, field_only 96x |
| `gymnasium.utils.env_checker` | passes (incl. reset-seed determinism) |
| Reset hygiene (V3) | same seed → same initial observation across episodes |
| Action plumbing (V4) | a different ration measurably changes the trajectory |
| RuFaS source (V2) | untouched |

### Scope and caveats — read before using a result

- **Only the ration lever is wired.** It is the only one with a passing Axis-B
  equivalence spike. Crop, fertilizer and manure schedules are fixed for the episode
  at construction, so the cross-*year* field dynamics the project is ultimately after
  are not yet controllable. `EnvConfig` refuses the unproven levers rather than
  silently accepting them.
- **The default reward is a bring-up proxy**, not the paper's objective.
  `MilkMinusNitrogen` prices milk against nitrate runoff and ignores feed cost, herd
  economics and greenhouse gases entirely. The real objective (EEE economics + enteric
  methane) is reported *annually*, so a faithful profit reward is sparse at monthly
  cadence — deliberately left for later, behind the same interface.
- **Observations come from live engine objects, not `variables_pool`.** The pool does
  accumulate during a run, but most soil variables are written by *annual* reporters,
  so at a monthly pause they lag by up to eleven months and would be blind to the
  agent's own recent actions. Reading the live object graph is inspection only.
- **`field_only` scenarios pause but have no lever.** Pausing is verified on Kimberly
  (96 pauses); the ration lever is meaningless without a herd, so those scenarios are
  observable but not yet actionable.
- **`spawn` requires an `if __name__ == "__main__":` guard** in any script that
  constructs an env, since each episode is a fresh interpreter.
- **Two RuFaS interactions worth knowing about, both handled.** (1) Abandoning an
  episode early unwinds through RuFaS's call stack, and `TaskManager`'s broad
  `except Exception` treated that as a *task failure*, dumping ~370 KB of logs per
  episode regardless of `suppress_log_files`; the teardown signal is therefore a
  `BaseException`, which passes straight through. (2) Per-task settings override the
  corresponding `TaskManager.start` kwargs, so `suppress_log_files` has to be set in the
  task JSON — otherwise every episode writes a ~470 KB metadata dump. Net result: an
  episode now writes nothing to disk outside its own temp directory.
- The first interval's milk is under-counted: `herd_statistics` is unpopulated at day 0,
  so the opening month's production rate reads zero. Mitigated by trapezoidal
  integration; absent entirely at yearly cadence.

## Step 0 — feasibility gate (V1)

Measured on the default `example_freestall` scenario (7 sim-years, 2013–2019),
in-process, `workers=1`, via `scripts/step0_timing.py`
(run with the RuFaS venv: `../RuFaS/venv/bin/python scripts/step0_timing.py`).

| Metric | Value |
|---|---|
| Full 7-year run (cold, fresh process) | **~26 s** |
| Per-simulated-year | **~3.8 s** |
| Strategy band | **seconds → O(T) online stepping + subprocess parallelism** |

### Three findings that shape the architecture

1. **~3.8 s/sim-year → "seconds" band.** Replay's O(T²) is *fatal* at the target
   **monthly cadence over ~10 years** (T≈120): ~60× the sim-work of the O(T)
   stepping patch. So replay is demoted from the default training path to a
   scaffold / correctness oracle only (V7). The training path is **O(T) online
   pause-and-resume** — primary: the zero-edit threaded pause hook; fallback: a
   flag-gated source patch (Axis A below).

2. **Two independent axes to the stepping problem** (verified in source):
   - **Axis A — pause mechanism.** One daily-granularity seam with a cadence
     predicate unifies yearly and monthly (yearly = coarse case). Options: generator
     source-patch, zero-edit threaded pause hook, or replay re-run. Deferred behind a
     `Stepper` interface.
   - **Axis B — action application at the pause (the hard, unproven part).** Every
     lever caches its config at **construction**: rations in `RationManager` class
     attrs (`herd_manager.py:191`), field ops in per-`Field` event lists
     (`field_manager.py:150+`). `formulate_rations`/`annual_update_routine` never
     re-read the pool. So **mid-run pool injection is a no-op** — the only proven
     injection is at construction (what replay uses). True online stepping requires
     **live-object mutation** of RuFaS internals for *every* lever, with a per-lever
     equivalence spike (does a mutated stepped run reproduce a construction-injected
     monolithic run?). The threaded hook dodges the *source edit*, not this risk.
   **Axis-B spike result (rations) — PASS** (`scripts/spike_axisb_rations.py`):
   live-mutating `RationManager` mid-run is physically identical to construction
   injection (0 physical keys differ vs a construction-injected run; 941 differ vs
   baseline, so the change bites). ⇒ **O(T) online stepping of the ration lever is
   viable.**
   **Axis-A spike result (pause mechanism) — PASS** (`scripts/spike_axisa_pause.py`):
   the zero-edit **threaded pause hook** — run RuFaS on a worker thread, monkey-patch
   `SimulationEngine._execute_full_farm_daily_simulation` (`:322`) to block on a
   hand-off queue at a cadence predicate (`current_date.day == 1` == monthly) — pauses
   the 7-year `example_freestall` run **84×** (7×12 monthly boundaries) and resumes to
   completion with **no deadlock**; its `variables_pool` is **byte-identical to an
   unhooked run (0 of 2702 physical keys differ)**, so pausing does not touch the
   science; and the STOP sentinel tears the worker down cleanly (the `env.reset()`
   path). ⇒ **Axis A proven with zero RuFaS edits; ready to wrap as the
   `ThreadedPauseStepper` (T7).**
   **Axis A + B integration — PASS** (`scripts/spike_pause_and_mutate.py`): changing a
   value *while the sim is genuinely suspended at a pause boundary* (driver live-mutates
   `RationManager` at pause #1, worker blocked) both **bites** (941 physical keys differ
   vs a no-op stepped run — the same 941 as the Axis-B result) and is **correct**
   (0 keys differ vs construction injection). ⇒ **`step(action)` semantics are sound:
   pause → apply action → resume == native input.**
   Still open: the same Axis-B spike for **field-op levers** (crop/fertilizer/manure
   at year boundaries — the thesis levers) and a later-boundary replay-equivalence
   check.

3. **In-process reuse degrades ~5×/run; cause unconfirmed.** A second run in the
   same interpreter is **~5× slower** than the first (148 s vs 27 s) while
   producing an *identical* 2845-key `variables_pool` — same work, 5× the time —
   *even after* flushing both the `InputManager` and `OutputManager` singleton
   pools and using a fresh output directory. Cause not pinned down (in-memory
   accumulation is likely but unconfirmed vs. re-init overhead); pinning it isn't
   necessary. RuFaS's own task manager already sidesteps this with
   `maxtasksperchild=1` for `workers>1`. **Consequence:** the harness must run each
   RuFaS invocation in a **fresh subprocess**, not reuse an in-process env — which
   promotes subprocess isolation from a Phase-3 add-on (T10) to a Phase-1
   foundation (T1). Note this degradation is *cross-episode* only; a single episode
   is one `start()` in one process, so it does **not** threaten the stepping patch
   itself.

### Cost implication for training — online is tractable at monthly cadence

The load-bearing number is the **cold full-horizon run (~26 s)**, since a
subprocess-per-episode model pays Python import + input/cross-validation JSON load
*every* episode; the 3.8 s/yr figure conflates that fixed startup with marginal
per-year cost (not yet split — a 1-year vs 7-year run would separate them).

Count in **transitions, not episodes**: with the stepping patch at monthly cadence,
one ~26 s run pauses ~120 times and yields ~120 `(s,a,r,s')` transitions (one
on-policy trajectory). So a ~1e6-transition budget ≈ 8,300 runs ≈ ~60 h
single-threaded → **~7–8 h with ~8 parallel subprocesses** (a 1e5 budget is under an
hour). Monthly cadence is therefore *efficient* per unit sim-time — more transitions
per expensive run. **Online PPO/SAC is the primary training mode.**

**Offline / growing-batch RL** (sample trajectories in parallel, log `(s,a,r,s')`,
train IQL/CQL off-policy) is a **fallback lever** if online wall-clock disappoints —
NOT a substitute for the simulator. Its dataset is still *simulator* counterfactual
trajectories; real farm data cannot supply the counterfactuals/coverage the project
depends on.

> Numbers are cold-cache wall-clock on the dev Mac; treat as order-of-magnitude.
