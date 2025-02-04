#!/bin/bash
#SBATCH --job-name=open-r1-dist-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod 
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

# Get the node name where the job is running
node_name=$(hostname)

# Resolve the IP address using the node name
node_ip=$(scontrol show node=$node_name | awk -F= '/NodeAddr=/ {print $2}')
node_ip=$(echo $node_ip | awk '{print $1}')
# Use SLURM_JOB_ID in the output file name
echo $node_ip > node_info_$SLURM_JOB_ID.txt
echo $node_name >> node_info_$SLURM_JOB_ID.txt

# Execute the command passed as the first argument ($1)
echo "Executing command: $1"
bash -c "$1"