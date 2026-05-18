#!/bin/bash
set -eo pipefail

# bash scripts/train_rl_policy.sh ${alg_name} ${task_name} ${setting} ${expert_data_num} ${addition_info} ${seed} ${gpu_id} ${pretrained_ckpt} [task_config]
#
# Runtime switches:
#   TRAIN=false EVAL=true bash scripts/train_rl_policy.sh ...
#   EVAL=true EXPORT_RL_RESUME_ACTOR=true bash scripts/train_rl_policy.sh ...

DEBUG=${DEBUG:-False}
train=false
eval=true

alg_name=${1:-reinflow_rl_pointcloud_robotwin2}
task_name=${2}
setting=${3}
expert_data_num=${4}
addition_info=${5}
seed=${6}
gpu_id=${7}
pretrained_ckpt=${8}
task_config=${9:-demo_clean_ur5}

is_blank() {
    [ -z "${1//[[:space:]]/}" ]
}

if is_blank "${alg_name}" || is_blank "${task_name}" || is_blank "${setting}" || \
   is_blank "${expert_data_num}" || is_blank "${addition_info}" || \
   is_blank "${seed}" || is_blank "${gpu_id}"; then
    echo -e "\033[31mMissing or blank required arguments.\033[0m"
    echo "Usage: bash scripts/train_rl_policy.sh reinflow_rl_pointcloud_robotwin2 <task> <setting> <expert_data_num> <addition> <seed> <gpu_id> <pretrained_ckpt> [task_config]"
    exit 1
fi

if ! [[ "${expert_data_num}" =~ ^[0-9]+$ ]]; then
    echo -e "\033[31mInvalid expert_data_num: ${expert_data_num}. Positional arguments may be shifted.\033[0m"
    exit 1
fi

if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo -e "\033[31mInvalid seed: ${seed}. Positional arguments may be shifted.\033[0m"
    exit 1
fi

if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo -e "\033[31mInvalid gpu_id: ${gpu_id}. Positional arguments may be shifted.\033[0m"
    exit 1
fi

config_name=${alg_name}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="/media/Elements1/ljj/ManiFlow/outputs/${exp_name}_seed${seed}"
policy_name=ManiFlow
eval_task_config=${EVAL_TASK_CONFIG:-${task_config}}
eval_ckpt_setting=${EVAL_CKPT_SETTING:-${task_config}}
eval_seed=${EVAL_SEED:-0}
# 导出RL resume actor的开关，默认为true，如果设置为false，则eval_ckpt_tag默认为latest
export_rl_resume_actor=false
if [ -n "${EVAL_CKPT_TAG:-}" ]; then
    eval_ckpt_tag=${EVAL_CKPT_TAG}
elif [ "${export_rl_resume_actor}" = true ]; then
    eval_ckpt_tag=final_rl_actor
else
    eval_ckpt_tag=latest
fi
rl_resume_ckpt=${RL_RESUME_CKPT:-${run_dir}/checkpoints/latest_rl.ckpt}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFLOW_POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_DIR="${MANIFLOW_POLICY_DIR}/ManiFlow/maniflow/workspace"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUROBO_TORCH_COMPILE=${CUROBO_TORCH_COMPILE:-0}
export CUROBO_TORCH_COMPILE_DISABLE=${CUROBO_TORCH_COMPILE_DISABLE:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== RL DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== RL TRAINING MODE ===\033[0m"
fi

echo -e "\033[36m=== Parsed RL script args ===\033[0m"
echo "alg_name=${alg_name}"
echo "task_name=${task_name}"
echo "setting=${setting}"
echo "expert_data_num=${expert_data_num}"
echo "addition_info=${addition_info}"
echo "seed=${seed}"
echo "gpu_id=${gpu_id}"
echo "task_config=${task_config}"
echo "run_dir=${run_dir}"

if [ "${train}" = true ] && [ -z "${pretrained_ckpt}" ]; then
    echo -e "\033[31mMissing pretrained checkpoint path.\033[0m"
    echo "Usage: bash scripts/train_rl_policy.sh reinflow_rl_pointcloud_robotwin2 <task> <setting> <expert_data_num> <addition> <seed> <gpu_id> <pretrained_ckpt> [task_config]"
    exit 1
fi

if [ "${train}" = true ]; then
    cd "${WORKSPACE_DIR}"

    echo -e "\033[32m=== Starting ManiFlow PPO Fine-tuning ===\033[0m"
    ${PYTHON_BIN} train_reinflow_rl_robotwin2_workspace.py \
        --config-name=${config_name}.yaml \
        task_name=${task_name} \
        task_config=${task_config} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        logging.name=${exp_name} \
        expert_data_num=${expert_data_num} \
        setting=${setting} \
        rl.pretrained_checkpoint_path=${pretrained_ckpt}
else
    echo -e "\033[33m=== RL training disabled ===\033[0m"
fi

if [ "${eval}" = false ]; then
    echo -e "\033[33m=== Evaluation disabled ===\033[0m"
    exit 0
fi

if [ "${export_rl_resume_actor}" = true ]; then
    if [ ! -f "${rl_resume_ckpt}" ]; then
        echo -e "\033[31mRL resume checkpoint not found: ${rl_resume_ckpt}\033[0m"
        exit 1
    fi

    cd "${WORKSPACE_DIR}"
    echo -e "\033[32m=== Exporting deploy actor from RL resume checkpoint ===\033[0m"
    ${PYTHON_BIN} train_reinflow_rl_robotwin2_workspace.py \
        --config-name=${config_name}.yaml \
        task_name=${task_name} \
        task_config=${task_config} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device=cuda:0 \
        exp_name=${exp_name} \
        logging.mode=disabled \
        logging.name=${exp_name} \
        expert_data_num=${expert_data_num} \
        setting=${setting} \
        rl.resume_path=${rl_resume_ckpt} \
        rl.export_resume_actor_only=true \
        rl.export_resume_actor_tag=${eval_ckpt_tag} 


    exported_actor_ckpt="${run_dir}/checkpoints/${eval_ckpt_tag}.ckpt"
    if [ ! -f "${exported_actor_ckpt}" ]; then
        echo -e "\033[31mFailed to export deploy actor checkpoint: ${exported_actor_ckpt}\033[0m"
        exit 1
    fi
fi

echo -e "\033[32m=== Evaluating ManiFlow RL policy ===\033[0m"
echo -e "\033[33mckpt tag: ${eval_ckpt_tag}, gpu id: ${gpu_id}\033[0m"

cd "${ROBOTWIN_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --config_name ${config_name} \
    --task_name ${task_name} \
    --task_config ${eval_task_config} \
    --ckpt_setting ${eval_ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --training_seed ${seed} \
    --seed ${eval_seed} \
    --policy_name ${policy_name} \
    --addition_info ${addition_info} \
    --alg_name ${alg_name} \
    --ckpt_tag ${eval_ckpt_tag}
