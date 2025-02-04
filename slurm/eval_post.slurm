
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

# Function to get commit ID
get_commit_id() {
    python3 <<EOF
import re
from huggingface_hub import list_repo_commits

def step2rev(model_id, step):
    commits = list_repo_commits(model_id)
    ckpts = [c for c in commits if "step" in c.title]
    for c in ckpts:
        match = re.search(r"(step\s\d+)", c.title)
        if match and step == match.group().split(" ")[-1]:
            return c.commit_id
    return None  # Return None if not found

model_id = "$model_path"
step = "$1"
commit_id = step2rev(model_id, step)
print(commit_id if commit_id else "None")
EOF
}

get_commit_id_eot() {
    python3 <<EOF
import re
from huggingface_hub import list_repo_commits

def get_final_commit(model_id):
    commits = list_repo_commits(model_id)
    for c in commits:
        if "End of training" in c.title:
            return c.commit_id
    return None  # Return None if not found

model_id = "$model_path"
commit_id = get_final_commit(model_id)
print(commit_id if commit_id else "None")
EOF


steps=(100,200,300,400)
# Loop over datasets, splits, and steps
for dataset_name in "${datasets[@]}"; do
    for dataset_split in "${dataset_splits[@]}"; do
        for step in "${steps[@]}"; do
            commit_id=$(get_commit_id "$step")
            echo "Commit ID: $commit_id"

            if [ "$commit_id" == "None" ]; then
                echo "Warning: No commit found for step $step, skipping..."
                continue
            fi

            job_name="eval-${model_path//\//-}-$dataset_split-$step"  # Replace '/' with '-'

            echo "Submitting job: $job_name"
            sbatch --job-name="$job_name" --gres=gpu:$num_gpus evaluation/scripts/launch_eval_history.slurm \
                --model_path "$model_path" \
                --model_revision "$commit_id" \
                --column_name "$column_name" \
                --given_backtrack "$given_backtrack" \
                --dataset_name "$dataset_name" \
                --dataset_split "$dataset_split" \
                --env "$env" \
                --num_samples "$num_samples" \
                --temperature 0.7 \
                --max_tokens 4096
        done
        commit_id=$(get_commit_id_eot)
        echo "Commit ID: $commit_id"
        
    done
done