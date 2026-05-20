#!/usr/bin/env bash
# Generate 8 images for each variant in isolation (no '+' combinations).
#
# Override the defaults from the env if needed:
#   N=8                                  images per variant
#   OUT=samples/smoke                    output root
#   NPROC=1                              number of GPUs for torchrun
#   NUM_STEPS_SANA=4                     Sana inference steps
#   NUM_STEPS_DIT=25                     DiT inference steps
#   SPARSITY_PATH_SANA=...               required to run sparse_ffn on Sana
#   SPARSITY_PATH_DIT=...                required to run sparse_ffn on DiT
#   SANA_FFN_PLAN=...                    static plan; if unset, gsparse runs dynamic

set -euo pipefail

N="${N:-8}"
OUT="${OUT:-samples/smoke}"
NPROC="${NPROC:-1}"
NUM_STEPS_SANA="${NUM_STEPS_SANA:-4}"
NUM_STEPS_DIT="${NUM_STEPS_DIT:-25}"
SPARSITY_PATH_SANA="${SPARSITY_PATH_SANA:-}"
SPARSITY_PATH_DIT="${SPARSITY_PATH_DIT:-}"
SANA_FFN_PLAN="${SANA_FFN_PLAN:-}"

mkdir -p "$OUT"

run() {
    local model="$1" variant="$2"; shift 2
    local steps="$NUM_STEPS_SANA"
    [[ "$model" == "dit_xl" ]] && steps="$NUM_STEPS_DIT"

    local tag="${model}_${variant}_${steps}step"
    local out_dir="$OUT/$tag"

    if [[ -f "$out_dir/manifest.json" ]]; then
        echo "[skip] $tag already present"
        return 0
    fi

    echo "[run]  $tag  -> $out_dir"
    torchrun --nproc_per_node="$NPROC" -m dit_accel.cli.generate \
        model="$model" \
        dataset=imagenet \
        variant="$variant" \
        num_steps="$steps" \
        batch_size="$N" \
        limit="$N" \
        samples_per_class=1 \
        output_dir="$out_dir" \
        "$@"
}

# ---- Sana Sprint variants --------------------------------------------------
run sana_sprint bf16
run sana_sprint int8
run sana_sprint int8_bnb
run sana_sprint xattn
run sana_sprint lacache
run sana_sprint cached
run sana_sprint block_cache

if [[ -n "$SPARSITY_PATH_SANA" ]]; then
    run sana_sprint sparse_ffn sparsity_path="$SPARSITY_PATH_SANA"
else
    echo "[skip] sana_sprint sparse_ffn (set SPARSITY_PATH_SANA to enable)"
fi

if [[ -n "$SANA_FFN_PLAN" ]]; then
    run sana_sprint gsparse +sana_ffn=static sana_ffn.plan_path="$SANA_FFN_PLAN"
else
    run sana_sprint gsparse +sana_ffn=dynamic
fi

# ---- DiT-XL variants -------------------------------------------------------
run dit_xl bf16
run dit_xl int8
run dit_xl int8_bnb
run dit_xl block_cache

if [[ -n "$SPARSITY_PATH_DIT" ]]; then
    run dit_xl sparse_ffn sparsity_path="$SPARSITY_PATH_DIT"
else
    echo "[skip] dit_xl sparse_ffn (set SPARSITY_PATH_DIT to enable)"
fi

echo
echo "Done. Outputs under: $OUT"
