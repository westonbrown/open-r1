from datasets import load_dataset
from open_r1.dist_gpro.vllm_api_client import VllmApiClient
from openai import OpenAI


if __name__ == "__main__":


####################################################################
    # define a model we are going to generate from 
    # get the server url from a file or something
    server_url = "http://26.0.163.127:8000"
    server_ip = "26.0.163.127"
    server_port = "8000"
    vllm_client = VllmApiClient(server_ip, server_port)

    MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    vllm_client.launch_model(MODEL, 8123)
    vllm_client.wait_for_ready()
    vllm_client.check_health()
    vllm_client.get_status()


    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(10))

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }
    dataset = dataset.map(make_conversation)

        

    def process_result(result):
        all_logprobs = []
        all_responses = []
        
        for choice in result.choices:
            log_probs = [rec.logprob for rec in choice.logprobs.content]
            response = choice.message.content
            all_logprobs.append(log_probs)
            all_responses.append(response)
        
        return {
            "responses": all_responses,
            "logprobss": all_logprobs,
        }
        
        

    for d in dataset:
        print(d["prompt"])
    # Modify OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = f"{vllm_client.vllm_url}/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        result = client.chat.completions.create(
            messages=d["prompt"],
            model=MODEL,
            logprobs=True,
            max_tokens=32000,
            n=2,
            )
        processed_result = process_result(result)
        print("Completion result:", processed_result)


