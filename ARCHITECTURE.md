# Architecture — A Ground-Up Walkthrough

For someone new to the project. The formal spec is `RL_ENVIRONMENT_PLAN.md`; the build order is
`IMPLEMENTATION_ORDER.md`; the step-by-step coding assignment is `CODING_TASKS.md`. This file explains
*what we're building and why*.

## What exists before we start

**RuFaS** is a dairy-farm simulator. You hand it a complete description of a farm — how many cows,
what crops in which fields, the feeding plan, fertilizer and manure schedules, the weather — and it
simulates the farm day by day for several years. Out the other end come numbers: milk produced,
profit, greenhouse-gas emissions, nitrogen leaching into groundwater, soil condition. It is
*validated* — its outputs have been checked against real farms, so people trust its predictions. That
trust is the whole reason we build on it.

The catch: RuFaS runs **start to finish in one shot**. You give it everything up front, press go, and
wait for the full multi-year result. There is no "play one year, look around, then decide the next."

## What we want to build

An **AI agent that learns to manage the farm well** — choosing feed mixes, crop rotations, and
fertilizer amounts to maximize profit while keeping pollution down. This is a **reinforcement learning
(RL)** problem: an agent takes an action, sees what happens, gets a reward or penalty, and gradually
learns a strategy.

Why RL rather than trying every plan? Because farm decisions **echo across years**. Planting a legume
this year naturally fertilizes next year's corn. Over-applying manure boosts yield now but leaks
nitrogen years later. A good manager thinks ahead — and that "think ahead across coupled years" shape
is exactly what RL is built for.

## The core tension, and how we resolve it

RL needs an *environment* it can step through one decision at a time. RuFaS only runs all-at-once. So
the project is: **wrap RuFaS in a shell that makes it look like a step-by-step game — without changing
how RuFaS simulates the farm.**

The golden rule: we never touch the *physics* (how fertilizer affects soil, how feed affects milk).
Changing it would break RuFaS's validation and destroy the project's credibility. We may change the
*plumbing* (how we start and pause it), never the *science*.

## The pieces

The new code lives in its own package, **`rufas_rl/`**, separate from RuFaS. The centerpiece is a
class, **`RufasEnv`**, that follows **Gymnasium** — the standard interface every RL agent already
knows. Gymnasium environments expose two verbs:

- **`reset()`** — start a fresh episode (a clean farm).
- **`step(action)`** — apply one decision, return the new state + a reward.

Make RuFaS speak those two verbs and any off-the-shelf RL algorithm can train on it.

Inside, the work splits into four swappable helpers (a design borrowed from a similar project,
CyclesGym):

| Helper | Job | Plain English |
|---|---|---|
| **implementer** | action → RuFaS input | Translates the agent's numbers ("60% corn silage, 40 kg N fertilizer") into the config format RuFaS expects. |
| **observer** | RuFaS output → state | Reads RuFaS's large results blob and pulls out the numbers the agent should see (soil nitrogen, herd condition, etc.). |
| **rewarder** | RuFaS output → score | Turns results into a single reward — e.g. profit minus a pollution penalty. Swappable across objectives. |
| **constrainer** | RuFaS output → limits | Optional. Reports whether a hard limit (like a nitrogen cap) was violated, for "stay under this line" training. |

Underneath them sits the **harness** — the glue that launches RuFaS, feeds it the action, and
captures its output.

## How an action actually gets in

RuFaS loads its entire farm description into one big in-memory dictionary called the **pool** before
it starts simulating. RuFaS already has a side-channel — `input_patch` — for overlaying changes onto
that pool. We use it for the **first** decision of an episode: the harness injects the agent's
decision as a patch right before RuFaS builds the farm. No RuFaS files edited — we use a door the
simulator already has. (The sibling project `rufas-web` already does this exact trick, so it is
proven.)

**The catch (verified in the source): that door only works while the farm is being built.** During
construction each module copies its configuration *out* of the pool and never looks back — rations
end up cached on a class, field schedules become pre-built event lists on each field. Editing the
pool mid-run changes nothing. How decisions get in *after* the farm is running is the second half of
the stepping problem, below.

## The hard part: making it "step"

Since RuFaS runs all-at-once, "advance one period" is not native. We measured first: **Step 0 is
done** — one simulated year costs **~3.8 seconds** (~26 s for a full 7-year run). That number killed
the simplest idea, **replay** (to reach year 3, re-run years 0–3 from scratch every step): at our
target cadence of **monthly decisions over ~10 years** (~120 steps), replay re-simulates the whole
history at every step and does ~60× the work of pausing and resuming. So the design is
**pause-and-resume**, which splits into two separate problems:

- **Axis A — pausing.** Run RuFaS on a worker thread and wrap its daily-step method so that, at each
  decision boundary, the worker stops and waits on a queue until the agent answers. Zero RuFaS files
  edited — and the paused thread's own call stack *is* the saved farm state, so nothing needs
  serializing. Whether a boundary is "every year" or "every ~30 days" is just a predicate, so
  **yearly and monthly are one mechanism with a knob** — monthly is the primary target, yearly the
  coarse setting for bring-up and ablations. If the threading ever proves unmanageable, the fallback
  is a small flag-gated edit that turns RuFaS's loop into a pausable generator (switch off =
  byte-identical to the original).
- **Axis B — applying the decision at the pause (the part that needed proving).** The `input_patch`
  door above only works before the farm is built. So at each pause we re-run the exact setup function
  RuFaS itself used at construction, with the agent's new numbers, against the live farm objects —
  one small "applier" per lever. **Proven for rations** (`scripts/spike_axisb_rations.py`): a run
  whose ration was live-mutated mid-run is identical, in every physical output, to a run configured
  that way from the start. Crop/fertilizer/manure appliers are next, each gated by the same
  equivalence test.

**Replay survives in one crucial role: the referee.** A paused-mutated-resumed episode must reproduce
exactly what a single monolithic run produces when handed the same decisions up front. That test is
what proves the science stays untouched.

One more measured fact shapes the plumbing: a second RuFaS run *in the same process* comes out ~5×
slower than the first (something in RuFaS's global state degrades), so **every episode runs in its
own fresh subprocess** — the same medicine RuFaS's own task manager takes (`maxtasksperchild=1`).

## How it comes together, in order

1. **Phase 1 — prove the wiring.** The simplest version: one decision, one full simulation, one
   reward. Confirm that changing an action changes the outputs sensibly. Build two honest **baselines**
   (a greedy one-step optimizer and a fixed full-plan optimizer) to beat later.
2. **Phase 2 — make it sequential.** The agent decides period by period — **monthly is the target
   cadence** (yearly is the same knob turned coarse) — seeing the farm's real carryover state at each
   pause. Train an RL policy (PPO/SAC) **online** and show it beats both baselines by exploiting
   cross-period effects. **This is the paper's headline result.**
3. **Phase 3 — make it robust and fast.** Add random weather/price variation, average multiple runs to
   smooth out RuFaS's built-in noise, and run many episodes at once (one subprocess per episode is
   already the rule — this phase just runs many in parallel).

## The one-sentence version

We build a thin, standard-shaped RL wrapper around an unmodified validated dairy simulator, pause it
at each management decision with a zero-edit thread hook, apply the agent's decision through the
simulator's own setup functions (proven equivalent to native input), and let an agent learn
farm-management strategies that pay off across years — the paper's contribution being that framing
plus the trained policy.
