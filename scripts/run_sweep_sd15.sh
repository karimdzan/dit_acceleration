#!/usr/bin/env bash
set -euo pipefail

TEACHER_ID="${TEACHER_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
STUDENT_ID="${STUDENT_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
BASE_DIR="${BASE_DIR:-runs/sweep_sd15}"
SIZES=(${SIZES:-32 64 128 256 512})
STEPS="${STEPS:-600}"
LR="${LR:-1e-6}"
MP="${MP:-fp16}"
DTYPE="${DTYPE:-fp16}"
EVAL_SAMPLES="${EVAL_SAMPLES:-256}"
INF_STEPS="${INF_STEPS:-30}"
CFG="${CFG:-7.5}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "${BASE_DIR}"

for ((i=0; i<${#SIZES[@]}; i+=2)); do
  N0="${SIZES[$i]}"
  RUN0="${BASE_DIR}/N${N0}"
  mkdir -p "${RUN0}"

  echo "[GPU0] N=${N0}"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m scripts.run_experiment_sd15 \
    --run_dir "${RUN0}" \
    --teacher_id "${TEACHER_ID}" \
    --student_id "${STUDENT_ID}" \
    --train_size "${N0}" \
    --train_bs "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --mixed_precision "${MP}" \
    --train_steps "${STEPS}" \
    --lr "${LR}" \
    --num_inference_steps "${INF_STEPS}" \
    --guidance_scale "${CFG}" \
    --eval_samples "${EVAL_SAMPLES}" \
    > "${RUN0}/log.txt" 2>&1 &

  PID0=$!

  if (( i+1 < ${#SIZES[@]} )); then
    N1="${SIZES[$((i+1))]}"
    RUN1="${BASE_DIR}/N${N1}"
    mkdir -p "${RUN1}"

    echo "[GPU1] N=${N1}"
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u -m scripts.run_experiment_sd15 \
      --run_dir "${RUN1}" \
      --teacher_id "${TEACHER_ID}" \
      --student_id "${STUDENT_ID}" \
      --train_size "${N1}" \
      --dtype "${DTYPE}" \
      --train_bs "${BATCH_SIZE}" \
      --mixed_precision "${MP}" \
      --train_steps "${STEPS}" \
      --lr "${LR}" \
      --num_inference_steps "${INF_STEPS}" \
      --guidance_scale "${CFG}" \
      --eval_samples "${EVAL_SAMPLES}" \
      > "${RUN1}/log.txt" 2>&1 &

    PID1=$!
    wait "${PID0}" "${PID1}"
  else
    wait "${PID0}"
  fi
done

echo "Done: sweep results in ${BASE_DIR}"
