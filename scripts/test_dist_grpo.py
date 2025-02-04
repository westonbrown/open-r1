from open_r1.dist_gpro.slurm_job import SlurmJob
import time
# Example usage (assuming a shared filesystem)
job = SlurmJob(name="test-open-r1-dist-grpo",
                script_path="slurm/job.slurm",
                command="""
source openr1
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
""")

job_id = job.submit()
job.wait_for_running()
vllm_ip = job.node_ip

from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://{vllm_ip}:8000/v1"



for i in range(200):
    try:
        print(f"connceting to {openai_api_base}")
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        chat_response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
        )
        print("Chat response:", chat_response)
        
    except KeyboardInterrupt as e:
        break
    except Exception as e:
        print(e)
        
    time.sleep(10)



job.cancel()
print(f"Node IP: {job.node_ip}")
print(f"Node Name: {job.node_name}")