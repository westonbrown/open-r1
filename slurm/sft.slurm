#!/bin/bash
#SBATCH --job-name=open-r1-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --qos=high
#SBATCH --partition=hopper-prod 
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

set -x -e

# Save a copy of this script using the actual job name and ID
OUTPUT_SLURM_SCRIPT=./logs/slurm/
cp $0 "${OUTPUT_SLURM_SCRIPT}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurm"
echo "saved script $0 to ${OUTPUT_SLURM_SCRIPT}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurm"

source ~/.bashrc
conda activate openr1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

# MODEL_PATH=/fsx/elie_bakouch/open-r1/model/qwen-math-1.5B-32k
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
DATASET_PATH=open-r1/OpenThoughts-114k-math-H4-8k
RUN_NAME=LLAMA8B-GR1-math-8k
ACCELERATOR=zero3
grad_acc=2
seq_len=32768
LR=1e-5
batch_size=1
optim=adamw_torch
max_steps=500
#  if we need to save memory--optim {adamw_hf,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,grokadamw,schedule_free_adamw,schedule_free_sgd}
#   --adam_beta1 ADAM_BETA1, --adam-beta1 ADAM_BETA1
#                         Beta1 for AdamW optimizer (default: 0.9)
#   --adam_beta2 ADAM_BETA2, --adam-beta2 ADAM_BETA2
#                         Beta2 for AdamW optimizer (default: 0.999)
# adam-b1=0.9
# adam-b2=0.999
scheduler=linear
#  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}, --lr-scheduler-type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}
# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export CMD=" \
    src/open_r1/sft.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_PATH \
    --use_liger_kernel true \
    --lr_scheduler_type $scheduler \
    --learning_rate $LR \
    --warmup_ratio 0.1 \
    --optim $optim \
    --max_steps $max_steps \
    --packing \
    --max_seq_length $seq_len \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $grad_acc \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir data/$RUN_NAME \
    --push_to_hub \
    --hub_private_repo True \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 100 \
    --hub_model_id HuggingFaceH4/$RUN_NAME \
    --hub_strategy every_save
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $grad_acc \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
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

echo "Waiting 1 minute before cleanup..."
sleep 60

echo "Cleaning up data/$RUN_NAME..."
rm -rf data/$RUN_NAME


echo "END TIME: $(date)"