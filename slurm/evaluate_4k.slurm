#!/bin/bash
#SBATCH --job-name=open-r1-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod 
#SBATCH --time=01:59:00
#SBATCH --output=./logs/evaluate/%x-%j.out
#SBATCH --err=./logs/evaluate/%x-%j.err


set -x -e

# Save a copy of this script using the actual job name and ID
OUTPUT_SLURM_SCRIPT=./logs/evaluate/slurm/
cp $0 "${OUTPUT_SLURM_SCRIPT}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurm"
echo "saved script $0 to ${OUTPUT_SLURM_SCRIPT}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurm"

source ~/.bashrc
conda activate openr1
module load cuda/12.1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"


NUM_GPUS=8


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

MODELS=(
    # "/fsx/elie_bakouch/open-r1/model/qwen-1.5-r1"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "HuggingFaceH4/Qwen-Math-1.5B-Bespoke-ep3-3e5-128GBS"
    "/fsx/elie_bakouch/open-r1/data/bespoke-ep3-cosine-1e-4-optim-adamw_torch-4k"
    "/fsx/elie_bakouch/open-r1/data/bespoke-ep3-cosine-6e-5-optim-adamw_torch-4k"
    "/fsx/elie_bakouch/open-r1/data/bespoke-ep3-linear-1e-4-optim-adamw_torch-4k"
    "/fsx/elie_bakouch/open-r1/data/bespoke-ep3-linear-6e-5-optim-adamw_torch-4k"
    # "HuggingFaceH4/Qwen-Math-1.5B-Bespoke-ep3-3e5-128GBS"
)

OUTPUT_DIR=./data/4k-eval/
for MODEL in "${MODELS[@]}"
do
    MODEL_ARGS="pretrained=$MODEL,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=4096,gpu_memory_utilisation=0.8"
    lighteval vllm $MODEL_ARGS "custom|aime24|0|0,custom|math_500|0|0" \
        --custom-tasks src/open_r1/evaluate_4k.py \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --push-to-hub \
        --no-public-run \
        --results-org "HuggingFaceH4" \
        --output-dir $OUTPUT_DIR \
        --save-details



done
# DESTINATION="s3://fineweb-multilingual-v1/evals/elie-temp"
# JOB_ID=$SLURM_JOB_ID
# MODEL_FOLDER="${MODEL_FOLDER}-${JOB_ID}"
# DESTINATION="s3://fineweb-multilingual-v1/evals/elie-temp"
# echo "Folder already exists. Renamed to $MODEL_FOLDER"
# echo "syncing folder in $OUTPUT_DIR/$MODEL to $DESTINATION"
# s5cmd s5cmd sync --no-delete "$OUTPUT_DIR" "$DESTINATION"

echo "END TIME: $(date)"