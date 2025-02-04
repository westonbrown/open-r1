import tempfile
import subprocess
from huggingface_hub import list_repo_commits
import re
from typing import Optional, List
import os


def step2rev(model_id: str, step: str) -> Optional[str]:
    """Get commit ID for a specific training step."""
    commits = list_repo_commits(model_id)
    # First filter commits that contain "step"
    ckpts = [c for c in commits if "step" in c.title.lower()]
    for c in ckpts:
        # Look for "step X" pattern and extract the number
        match = re.search(r"(step\s+\d+)", c.title.lower())
        if match and str(step) == match.group().split()[-1]:  # Convert step to string for comparison
            return c.commit_id
    return None

def get_final_commit(model_id: str) -> Optional[str]:
    """Get commit ID for end of training."""
    commits = list_repo_commits(model_id)
    for c in commits:
        if "End of training" in c.title:
            return c.commit_id
    return None

def launch_slurm_job(launch_file_contents: str, *args) -> str:
    """Launch a SLURM job with given script contents and arguments."""
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]

def launch_evaluation_jobs(
    model_paths: List[str],
    datasets: list[str],
    steps: list[str],
    env: str,
    num_samples: int,
    num_gpus: int = 8,
    tensor_parallel: bool = False,
    trust_remote_code: bool = True,
    evaluate_eot: bool = True,
    generation_size: int = 32768,
    qos: str = "normal",
    # Path parameters
    logs_dir: str = "./logs/evaluate",
    slurm_scripts_dir: str = "./logs/evaluate/slurm",
    eval_output_dir: str = "./data/eval",
    custom_tasks_path: str = "src/open_r1/evaluate.py",
    details_upload_script: str = "src/open_r1/utils/upload_details.py",
    lm_eval_repo_id: str = "open-r1/open-r1-eval-leaderboard",
    details_repo_base: str = "open-r1/details",
    extract_evals_script: str = "src/open_r1/utils/extract_evals.py",
    results_csv_path: str = "./data/eval/results.csv"
):
    """Launch evaluation jobs for different model checkpoints.
    
    Args:
        model_paths: List of model paths to evaluate
        datasets: List of dataset names
        steps: List of training steps to evaluate
        env: Conda environment name
        num_samples: Number of samples for evaluation
        num_gpus: Number of GPUs to use
        tensor_parallel: Whether to use tensor parallelism
        trust_remote_code: Whether to trust remote code
        evaluate_eot: Whether to evaluate default model version
        generation_size: Maximum model length for default model version
        qos: Quality of Service for SLURM job
        logs_dir: Directory for SLURM logs
        slurm_scripts_dir: Directory for saving SLURM scripts
        eval_output_dir: Directory for evaluation outputs
        custom_tasks_path: Path to custom tasks script
        details_upload_script: Path to details upload script
        lm_eval_repo_id: HuggingFace repo ID for leaderboard
        details_repo_base: Base repo ID for details
        extract_evals_script: Path to extract_evals.py script
        results_csv_path: Path to results CSV file
    """
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(slurm_scripts_dir, exist_ok=True)
    
    base_slurm_script = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --partition=hopper-prod 
#SBATCH --qos={qos}
#SBATCH --time=01:59:00
#SBATCH --output={logs_dir}/%x-%j.out
#SBATCH --err={logs_dir}/%x-%j.err
#SBATCH --requeue

set -x -e

# Save a copy of this script using the actual job name and ID
OUTPUT_SLURM_SCRIPT="{slurm_scripts_dir}"
cp $0 "$OUTPUT_SLURM_SCRIPT/$SLURM_JOB_NAME-$SLURM_JOB_ID.slurm"
echo "saved script $0 to $OUTPUT_SLURM_SCRIPT/$SLURM_JOB_NAME-$SLURM_JOB_ID.slurm"

# Define output directory
OUTPUT_DIR="{eval_output_dir}/{model_name}/{commit_id}"
mkdir -p $OUTPUT_DIR

source ~/.bashrc
conda activate {env}
module load cuda/12.1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

NUM_GPUS={num_gpus}

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_USE_DEEPSPEED=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

lighteval vllm {model_args} "{dataset_string}" \\
    --custom-tasks {custom_tasks_path} \\
    --use-chat-template \\
    --system-prompt="Please reason step by step, and put your final answer within \\boxed{{}}." \\
    --push-to-hub \\
    --no-public-run \\
    --results-org "HuggingFaceH4" \\
    --output-dir $OUTPUT_DIR \\
    --save-details

# Extract results to local CSV
OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f \( -name "*.json" \))
for filepath in $OUTPUT_FILEPATHS; do
    echo "Extracting results from $filepath to CSV..."
    python {extract_evals_script} \
        --results_file $filepath \
        --csv_file {results_csv_path}
done

# Cleanup
echo "Cleaning up..."
rm -rf $OUTPUT_DIR

echo "END TIME: $(date)"
"""

    for model_path in model_paths:
        model_name = model_path.split("/")[-1]
        jobs_submitted = False
        
        # Process regular steps
        for step in steps:
            commit_id = step2rev(model_path, step)
            if not commit_id:
                print(f"Warning: No commit found for step {step} in {model_path}")
                continue

            # Construct model_args string
            parallel_arg = f"tensor_parallel_size={num_gpus}" if tensor_parallel else f"data_parallel_size={num_gpus}"
            model_args = f"pretrained={model_path},revision={commit_id},trust_remote_code={str(trust_remote_code).lower()},dtype=bfloat16,{parallel_arg},max_model_length={generation_size},gpu_memory_utilisation=0.8"

            # Create dataset string with all datasets
            dataset_string = ",".join([f"custom|{dataset}|0|0" for dataset in datasets])

            job_name = f"eval-{model_path.replace('/', '-')}-step{step}"
            script_contents = base_slurm_script.format(
                num_gpus=num_gpus,
                qos=qos,
                logs_dir=logs_dir,
                slurm_scripts_dir=slurm_scripts_dir,
                env=env,
                custom_tasks_path=custom_tasks_path,
                lm_eval_repo_id=lm_eval_repo_id,
                details_upload_script=details_upload_script,
                model_args=model_args,
                dataset_string=dataset_string,
                commit_id=commit_id,
                eval_output_dir=eval_output_dir,
                model_name=model_name,
                extract_evals_script=extract_evals_script,
                results_csv_path=results_csv_path
            )
            job_id = launch_slurm_job(script_contents, "--job-name", job_name)
            print(f"Submitted job {job_name} with ID: {job_id}")
            jobs_submitted = True

        # Process default model version if enabled
        if evaluate_eot:
            # Construct model_args string without revision to use default version
            parallel_arg = f"tensor_parallel_size={num_gpus}" if tensor_parallel else f"data_parallel_size={num_gpus}"
            model_args = f"pretrained={model_path},trust_remote_code={str(trust_remote_code).lower()},dtype=bfloat16,{parallel_arg},max_model_length={generation_size},gpu_memory_utilisation=0.8"

            # Create dataset string with all datasets
            dataset_string = ",".join([f"custom|{dataset}|0|0" for dataset in datasets])

            job_name = f"eval-{model_path.replace('/', '-')}-default"
            script_contents = base_slurm_script.format(
                num_gpus=num_gpus,
                qos=qos,
                logs_dir=logs_dir,
                slurm_scripts_dir=slurm_scripts_dir,
                env=env,
                custom_tasks_path=custom_tasks_path,
                lm_eval_repo_id=lm_eval_repo_id,
                details_upload_script=details_upload_script,
                model_args=model_args,
                dataset_string=dataset_string,
                commit_id="default",
                eval_output_dir=eval_output_dir,
                model_name=model_name,
                extract_evals_script=extract_evals_script,
                results_csv_path=results_csv_path
            )
            job_id = launch_slurm_job(script_contents, "--job-name", job_name)
            print(f"Submitted default model job {job_name} with ID: {job_id}")
            jobs_submitted = True
        
        if not jobs_submitted:
            print(f"Warning: No jobs submitted for {model_path}")


if __name__ == "__main__":
    launch_evaluation_jobs(
        model_paths=[
            "Qwen/Qwen2.5-Math-1.5B-Instruct"
        ],
        datasets=["aime24", "math_500"],
        steps=[],
        env="openr1",
        num_samples=100,
        num_gpus=8,
        tensor_parallel=False,
        trust_remote_code=True,
        evaluate_eot=True,
        # Optional path configurations
        qos="high",
        logs_dir="./logs/evaluate/_REDO-QWENMATH7B-INSTRUCT",
        slurm_scripts_dir="./logs/evaluate/slurm",
        eval_output_dir="./data/eval",
        generation_size=4096,
        custom_tasks_path="src/open_r1/evaluate_4k.py",
        details_upload_script="src/open_r1/utils/upload_details.py",
        lm_eval_repo_id="open-r1/open-r1-eval-leaderboard",
        details_repo_base="open-r1/details",
        extract_evals_script="src/open_r1/utils/extract_evals.py",
        results_csv_path="./eval_csv/results_REDO-QWENMATH7B-INSTRUCT.csv"
    )
