import json
from typing import Optional, Dict
from openai import OpenAI
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch

# Load API keys from secrets.txt
SECRET_FILE = 'secrets.txt'
openai_key = None

try:
    with open(SECRET_FILE) as f:
        lines = f.readlines()
        for line in lines:
            if line.split(',')[0].strip() == "openai_key":
                openai_key = line.split(',')[1].strip()
except FileNotFoundError:
    print(f"Warning: {SECRET_FILE} not found. API calls will fail.")

# Initialize API clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None

# Cache for model instances
_model_cache: Dict[str, tuple] = {}  # (model, tokenizer) for HF models

def call_huggingface_model(prompt: Optional[str] = None, model_name: str = "microsoft/Phi-4-mini-instruct", model_tokenizer: tuple = None) -> Optional[str]:
    """Call a Hugging Face model for text generation (memory-efficient version using float16 on cuda:0)."""
    try:
        if model_tokenizer is None:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Load model in float16 and move to cuda:0
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to("cuda:0")
        else:
            model, tokenizer = model_tokenizer

        # Construct prompt
        messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize and move input to cuda
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
        )

        # Decode output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "llama" in model_name.lower():
            split_token = "assistant\n\n"
            if split_token in full_output.lower():
                idx = full_output.lower().rfind(split_token)
                full_output = full_output[idx + len(split_token):].strip()

            result = full_output
        else: 
            full_output = re.sub(r'^[Aa]ssistant\s*\n+', '', full_output, count=1).strip()

            # Remove prompt from output if duplicated
            response = full_output.replace(messages[0]["content"], "").strip()
            result = response if response else full_output.strip()

        return result

    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return None



def call_api_model(prompt: Optional[str] = None, model: str = "gpt-4o-mini") -> Optional[str]:
    """Call an API-based model for text generation.
    
    Args:
        prompt: Optional prompt to set the context
        model: The model to use (e.g., "gpt-4o-mini")
    
    Returns:
        The model's response text or None if there was an error
    """
    if not openai_client:
        print("Error: OpenAI client not initialized. Check your API key.")
        return None
        
    try:
        messages = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling {model}: {e}")
        return None 

def create_model_caller(model_name: str = "gpt-4o-mini"):
    """Create a model caller function with the specified model
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        A function that takes (prompt, input_text) and returns the model's response
    """
    # Check if the model is a Hugging Face model
    if "/" in model_name:  # Hugging Face models typically have a "/" in their name
        # Check if model is already in cache
        if model_name not in _model_cache:
            # print(f"Loading model {model_name} into cache...")
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            # Load model in float16 and move to cuda:0
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to("cuda:0")
            _model_cache[model_name] = (model, tokenizer)
            # print(f"Model {model_name} loaded and cached.")
        # else:
        #     print(f"Using cached model {model_name}")
        
        def model_caller(prompt: str) -> str:
            return call_huggingface_model(prompt=prompt, model_name=model_name, model_tokenizer=_model_cache[model_name])
    else:  # Assume it's an API model
        def model_caller(prompt: str) -> str:
            return call_api_model(prompt=prompt, model=model_name)
    
    return model_caller 