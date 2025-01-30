#!/bin/bash
#SBATCH --job-name=open-r1-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --priority=high
#SBATCH --partition=hopper-prod 
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

set -x -e

source ~/.bashrc
conda activate openr1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

MODEL_PATH=$1
DATASET_PATH=$2
ACCELERATOR=$3

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))


# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export CMD=" \
    src/open_r1/grpo.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_PATH \
    --learning_rate 1.0e-6 \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --max_prompt_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --report_to wandb \
    --logging_steps 1 \
    --eval_strategy no \
    --output_dir data/Qwen2.5-1.5B-Open-R1-GRPO-v0.1 \
    --hub_model_id open-r1/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-v0.1 \
    --save_strategy steps \
    --save_steps 10 \
    --callbacks push_to_hub_revision \
    --benchmarks math_500
    "
# accelerate launch --config_file configs/zero3.yaml      src/open_r1/grpo.py     --output_dir data/train/GPRO/DeepSeek-R1-Distill-Qwen-1.5B-v0.0     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     --dataset_name HuggingFaceH4/MATH     --max_prompt_length 512     --max_completion_length 1024     --per_device_train_batch_size 1     --gradient_accumulation_steps 4     --learning_rate 1e-6     --logging_steps 1     --bf16     --report_to wandb     --hub_model_id open-r1/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-v0.0     --save_strategy steps     --save_steps 10     --eval_strategy no     --callbacks push_to_hub_revision     --benchmarks math_500

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file configs/$ACCELERATOR.yaml  \
    --num_processes $(($WORLD_SIZE - 1)) \
    --gradient_accumulation_steps 4 \
    --num_machines $NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
