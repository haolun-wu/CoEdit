import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Union, Dict, List

class BaseLLM:
    def __init__(self, name: str):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.dummy = False

    def get_response_given_completion_prompt(
        self, prompt: str, temperature: float = 0.0, max_attempt=10000, max_tokens=300, expected_finish_reason="stop"
    ):
        if self.dummy:
            return prompt[:100]

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def get_response_given_chat_completion_prompt(
        self,
        chat_log: List[Dict],
        temperature: float = 0.0,
        max_attempt=10000,
        max_tokens=300,
        expected_finish_reason="stop",
    ):
        if self.dummy:
            return chat_log

        # Convert chat log to prompt
        prompt = ""
        for msg in chat_log:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"

        return self.get_response_given_completion_prompt(
            prompt, temperature, max_attempt, max_tokens, expected_finish_reason
        )

    def get_logprobs(self, prompt, temperature: float = 0.0, max_attempt=10000):
        return None

    def get_prompt_length(self, prompt, temperature: float = 0.0, max_attempt=10000):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return len(inputs["input_ids"][0]) 