#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: bash eval.sh CHECKPOINT TASK TASK_CONFIG [EVAL_SEED=10] [GPU_ID=0|visible] [--override value ...]" >&2
  exit 2
fi
FPOPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_DIR="$(cd "${FPOPP_DIR}/../.." && pwd)"
checkpoint_path="$(realpath "$1")"
task_name="$2"
task_config="$3"
eval_seed="${4:-10}"
gpu_id="${5:-0}"
if [ "$#" -ge 5 ]; then shift 5; else set --; fi
if [ "${gpu_id}" != visible ]; then export CUDA_VISIBLE_DEVICES="${gpu_id}"; fi
export PYTHONPATH="${FPOPP_DIR}:${ROBOTWIN_DIR}/policy/ManiFlow/ManiFlow:${ROBOTWIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-${ROBOTWIN_DIR}/policy/ManiFlow/scripts/nvidia_icd.json}"
export CUROBO_TORCH_COMPILE="${CUROBO_TORCH_COMPILE:-0}"
export CUROBO_TORCH_COMPILE_DISABLE="${CUROBO_TORCH_COMPILE_DISABLE:-1}"
cd "${ROBOTWIN_DIR}"
exec "${PYTHON_BIN:-python}" script/eval_policy.py --config policy/FPOpp/deploy_policy.yml \
  --overrides --policy_name FPOpp --checkpoint_path "${checkpoint_path}" \
  --task_name "${task_name}" --task_config "${task_config}" --ckpt_setting "${task_config}" \
  --seed "${eval_seed}" "$@"
