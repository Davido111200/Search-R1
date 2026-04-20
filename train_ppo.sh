#!/bin/bash
#SBATCH --job-name=train_ppo_search_r1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --output=output_training_job/%x_%j.out
#SBATCH --error=output_training_job/%x_%j.err
#SBATCH --exclude=rtxp6000l-f-01


set -euo pipefail

module load Anaconda3
source activate
conda activate searchr1_new

# Use the Search-R1 fork preserved in ./verl_old, not the upstream verl package
# installed in the conda environment.
VERL_IMPORT_ROOT=${VERL_IMPORT_ROOT:-$PWD/.ray_verl_import_${SLURM_JOB_ID:-local}}
mkdir -p "$VERL_IMPORT_ROOT"
ln -sfn "$PWD/verl_old" "$VERL_IMPORT_ROOT/verl"
export PYTHONPATH="$VERL_IMPORT_ROOT:$PWD:${PYTHONPATH:-}"

python3 - <<'PY'
import torch
import verl

print("Using verl from:", verl.__file__)
print("Using torch:", torch.__version__)
PY

# Slurm sets CUDA_VISIBLE_DEVICES for the GPUs allocated to this job.
export DATA_DIR='data/nq_search'

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
    NODE_LIST=("$(hostname)")
fi

NNODES=${#NODE_LIST[@]}
HEAD_NODE=${NODE_LIST[0]}
RAY_PORT=${RAY_PORT:-6379}
CPUS_PER_NODE=${SLURM_CPUS_PER_TASK:-8}

VISIBLE_GPUS_PER_TASK=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    VISIBLE_GPUS_PER_TASK=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi

if [[ -n "${VISIBLE_GPUS_PER_TASK}" ]]; then
    if [[ -n "${GPUS_PER_NODE:-}" && "${GPUS_PER_NODE}" != "${VISIBLE_GPUS_PER_TASK}" ]]; then
        echo "Ignoring GPUS_PER_NODE=${GPUS_PER_NODE}; CUDA_VISIBLE_DEVICES exposes ${VISIBLE_GPUS_PER_TASK} GPU(s)"
    fi
    GPUS_PER_NODE=${VISIBLE_GPUS_PER_TASK}
elif [[ -n "${GPUS_PER_NODE:-}" ]]; then
    :
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}
else
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
fi

echo "Slurm nodes: ${NODE_LIST[*]}"
echo "Head node: ${HEAD_NODE}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "CPUs per node: ${CPUS_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

export GLOO_SOCKET_IFNAME=bond0  # Gloo TCP transport interface for inter-node CPU groups

cleanup_ray() {
    echo "Stopping Ray cluster"
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        srun --nodes="${NNODES}" --ntasks="${NNODES}" bash -lc 'ray stop --force >/dev/null 2>&1 || true' || true
    else
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_ray EXIT

cleanup_ray

HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "hostname -I | awk '{print \$1}'")
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Starting Ray head at ${RAY_ADDRESS}"
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    bash -lc "RAY_NODE_GPUS=${GPUS_PER_NODE}; if [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" && \"\${CUDA_VISIBLE_DEVICES}\" != \"NoDevFiles\" ]]; then IFS=',' read -ra cuda_devices <<< \"\${CUDA_VISIBLE_DEVICES}\"; RAY_NODE_GPUS=\${#cuda_devices[@]}; fi; echo Ray head CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} RAY_NODE_GPUS=\${RAY_NODE_GPUS}; ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --dashboard-host=0.0.0.0 --num-gpus=\${RAY_NODE_GPUS} --num-cpus=${CPUS_PER_NODE} --block" &

sleep 10

if (( NNODES > 1 )); then
    for node in "${NODE_LIST[@]:1}"; do
        echo "Starting Ray worker on ${node}"
        srun --nodes=1 --ntasks=1 -w "${node}" \
            bash -lc "RAY_NODE_GPUS=${GPUS_PER_NODE}; if [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" && \"\${CUDA_VISIBLE_DEVICES}\" != \"NoDevFiles\" ]]; then IFS=',' read -ra cuda_devices <<< \"\${CUDA_VISIBLE_DEVICES}\"; RAY_NODE_GPUS=\${#cuda_devices[@]}; fi; echo Ray worker CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} RAY_NODE_GPUS=\${RAY_NODE_GPUS}; ray start --address=${RAY_ADDRESS} --num-gpus=\${RAY_NODE_GPUS} --num-cpus=${CPUS_PER_NODE} --block" &
    done
    sleep 20
fi

python3 - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"])
print("Ray cluster resources:", ray.cluster_resources())
print("Ray available resources:", ray.available_resources())
ray.shutdown()
PY

WAND_PROJECT='Search-R1'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
export BASE_MODEL='Qwen/Qwen3-1.7B-Base'
export EXPERIMENT_NAME=nq-search-r1-ppo-qwen3-1.7b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    +actor_rollout_ref.actor.autocast_dtype=fp16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=half \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    +critic.model.attn_implementation=sdpa \
    +critic.autocast_dtype=fp16 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    +critic.model.fsdp_config.mixed_precision.param_dtype=fp16 \
    +critic.model.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +critic.model.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://10.72.191.62:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
