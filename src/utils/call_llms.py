import json
from typing import Dict, Optional
from openai import OpenAI
from pathlib import Path

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


def call_llm(message: str, system_message: Optional[str] = None, model: str = "gpt-4o-mini") -> Optional[str]:
    """Call the specified LLM model for text information and return the response.
    
    Args:
        message: The user message to send to the model
        system_message: Optional system message to set the context
        model: The model to use (e.g., "gpt-4o-mini")
    
    Returns:
        The model's response text or None if there was an error
    """
    if not openai_client:
        print("Error: OpenAI client not initialized. Check your API key.")
        return None
        
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8000
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
    def model_caller(prompt: str, input_text: str) -> str:
        return call_llm(input_text, system_message=prompt, model=model_name)
    
    return model_caller 