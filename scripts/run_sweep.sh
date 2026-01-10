#!/usr/bin/env bash
set -euo pipefail

TEACHER_ID="${TEACHER_ID:-facebook/DiT-XL-2-256}"
BASE_DIR="${BASE_DIR:-runs/sweep}"
SIZES=(${SIZES:-32 64 128 256 512 1024})
STEPS="${STEPS:-600}"
LR="${LR:-1e-6}"
MP="${MP:-no}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
INF_STEPS="${INF_STEPS:-30}"
DESC="${DESC:-identity}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "${BASE_DIR}"

for ((i=0; i<${#SIZES[@]}; i+=2)); do
  N0="${SIZES[$i]}"
  RUN0="${BASE_DIR}/N${N0}"
  mkdir -p "${RUN0}"

  echo "[GPU0] N=${N0}"
  CUDA_VISIBLE_DEVICES=0 python -m scripts.run_experiment \
    --run_dir "${RUN0}" \
    --teacher_id "${TEACHER_ID}" \
    --train_size "${N0}" \
    --max_train_steps "${STEPS}" \
    --train_batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --mixed_precision no \
    --eval_samples "${EVAL_SAMPLES}" \
    --num_inference_steps "${INF_STEPS}" \
    --descriptor "${DESC}" \
    > "${RUN0}/log.txt" 2>&1 &

  PID0=$!

  if (( i+1 < ${#SIZES[@]} )); then
    N1="${SIZES[$((i+1))]}"
    RUN1="${BASE_DIR}/N${N1}"
    mkdir -p "${RUN1}"

    echo "[GPU1] N=${N1}"
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_experiment \
      --run_dir "${RUN1}" \
      --teacher_id "${TEACHER_ID}" \
      --train_size "${N1}" \
      --max_train_steps "${STEPS}" \
      --train_batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --mixed_precision no \
      --eval_samples "${EVAL_SAMPLES}" \
      --num_inference_steps "${INF_STEPS}" \
      --descriptor "${DESC}" \
      > "${RUN1}/log.txt" 2>&1 &

    PID1=$!
    wait "${PID0}" "${PID1}"
  else
    wait "${PID0}"
  fi
done

# Plot all results
python -m scripts.plot_sweep --results_glob "${BASE_DIR}/N*/metrics.json" --out_dir "${BASE_DIR}"
echo "Done: sweep + plots in ${BASE_DIR}"
