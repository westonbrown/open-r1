from datasets import load_dataset
from open_r1.dist_gpro.vllm_api_client import VllmApiClient
from openai import OpenAI
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer


class DistGRPOTrainer(Trainer):
    def process_result(self,result):
        all_logprobs = []
        all_responses = []
        
        for choice in result.choices:
            log_probs = choice.logprobs.token_logprobs
            response = choice.text
            all_logprobs.append(log_probs)
            all_responses.append(response)
        
        return {
            "responses": all_responses,
            "logprobss": all_logprobs,
        }
        
    def remote_generate(self, examples):
        convs = []
        for example in examples:
            convs.append([{
                "role": "user", 
                "content": example["problem"],
                }])
            
        prompts = self.processing_class.apply_chat_template([c for c in convs], tokenize=False, add_generation_prompt=True)
            
        openai_api_key = "EMPTY"
        openai_api_base = f"{self.policy_vllm_client.vllm_url}/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        # we need to be careful here with double BOS tokens, e.g llama3.2
        num_generations=2
        
        # we cannot batch with the chat api...
        result = client.completions.create(
            prompt=prompts,
            # echo=True,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            logprobs=True,
            max_tokens=32000,
            n=num_generations,   
            extra_body={
                "skip_special_tokens": "False",
            }
        )
        
        processed_result = self.process_result(result)
        processed_result["prompts"] = [[p]*num_generations for p in prompts]
        
        return processed_result
    
    def train(self):
        dataloader = self.get_train_dataloader()
        device = self.accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        
        for update in range(10):
            examples = next(iter_dataloader)
            # generate responses for these prompts
            generations = self.remote_generate(examples)
            # get the log probs 
            
            # get the reward
            
            
            pass


def build_client(model_name):
    server_ip = "26.0.163.127"
    server_port = "8000"
    vllm_client = VllmApiClient(server_ip, server_port)


    vllm_client.launch_model(model_name, 8123)
    vllm_client.wait_for_ready()
    vllm_client.check_health()
    vllm_client.get_status()
    
    return vllm_client


if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(256))

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }
    dataset.map(make_conversation)
    args = TrainingArguments(output_dir="data/test_dist_grpo_000", 
                             remove_unused_columns=False,
                             do_train=True, 
                             )
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    vllm_client = build_client(model_name)
    trainer = DistGRPOTrainer(
        model,
        args,
        train_dataset=dataset,
        data_collator=lambda x: x,
        processing_class=tokenizer,
    )
    trainer.policy_vllm_client = vllm_client
    trainer.ref_vllm_client = vllm_client
    
    trainer.train()




