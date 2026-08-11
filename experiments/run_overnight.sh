#!/usr/bin/env bash
#
# One-shot overnight PPO training run for RufasEnv.
#
# Runs from a clean, quiet machine — free up CPU first (the parallel envs fight anything
# else for cores). Trains, then auto-generates the learning-curve figure.
#
#   bash experiments/run_overnight.sh                 # defaults below
#   TIMESTEPS=50000 N_ENVS=6 bash experiments/run_overnight.sh   # override
#
# Watch it:  tail -f results/ppo_overnight/train.log
# It writes:  results/ppo_overnight/{ppo_rufas.zip, vecnormalize.pkl, result.json, train.log}
#             results/figures/5_ppo_learning.png
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- knobs (override via env) ------------------------------------------------
TIMESTEPS="${TIMESTEPS:-30000}"   # ~2.8 h at 4 envs on an idle machine
N_ENVS="${N_ENVS:-4}"             # 4 is the sweet spot; more oversubscribes cores
YEARS="${YEARS:-2}"               # 2-yr episodes train faster; 7 shows cross-year effects
SEED="${SEED:-0}"
OUT="${OUT:-results/ppo_overnight}"

# Pin every math library to one thread per process. Each of the N_ENVS parallel RuFaS
# simulations otherwise spawns its own BLAS/numba thread pool, oversubscribing the cores
# and thrashing (this is what stalled the interactive runs).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

mkdir -p "$OUT"
echo "PPO overnight run"
echo "  timesteps=$TIMESTEPS  n_envs=$N_ENVS  years=$YEARS  seed=$SEED"
echo "  output   =$OUT"
echo "  threads pinned to 1/process"
echo "  started  $(date)"
echo

# The stability settings (small learning rate, target_kl, low entropy bonus) live in
# train_ppo.py — they stop the policy-collapse seen in the first tuned run.
"$PY" experiments/train_ppo.py \
    --timesteps "$TIMESTEPS" \
    --n-envs "$N_ENVS" \
    --years "$YEARS" \
    --seed "$SEED" \
    --out "$OUT" 2>&1 | tee "$OUT/train.log"

echo
echo "Training done $(date). Building figure..."
"$PY" experiments/make_ppo_figure.py \
    --result "$OUT/result.json" \
    --out results/figures/5_ppo_learning.png

echo
echo "Done. Result: $OUT/result.json ; figure: results/figures/5_ppo_learning.png"
grep -E "learned policy|neutral policy|learned - neutral" "$OUT/train.log" || true
