from typing import List, Dict, Any, Set, Callable
import openai
import os
from dataclasses import dataclass
from enum import Enum
from .user_preferences import USER_PREFERENCES, GLOBAL_GUIDELINES, AtomicIntent

@dataclass
class IntentConfig:
    intents: Set[AtomicIntent]
    description: str
    prompt_template: str

class IntentHandler:
    def __init__(self, model_caller: Callable[[str, str], str]):
        """
        Initialize intent handler with a model caller function
        Args:
            model_caller: A function that takes (prompt, input_text) and returns the model's response
        """
        self.model_caller = model_caller
        self.intents: Dict[str, Dict[str, Dict[str, IntentConfig]]] = {}  # task -> dataset -> user_id -> intents
        self._initialize_intents()
        
    def _initialize_intents(self):
        """Initialize intents for each task, dataset, and user"""
        for user_id, user_pref in USER_PREFERENCES.items():
            for task, task_prefs in user_pref.task_preferences.items():
                if task not in self.intents:
                    self.intents[task] = {}
                
                for dataset, intents in task_prefs.items():
                    if dataset not in self.intents[task]:
                        self.intents[task][dataset] = {}
                    
                    description = self._get_intent_description(intents)
                    prompt_template = self._create_prompt_template(intents)
                    self.intents[task][dataset][user_id] = IntentConfig(intents, description, prompt_template)
    
    def _get_intent_description(self, intents: Set[AtomicIntent]) -> str:
        """Get description for a set of intents"""
        descriptions = []
        for intent in intents:
            if intent in AtomicIntent:
                descriptions.append(intent.value)
        return ", ".join(descriptions)
    
    def _create_prompt_template(self, intents: Set[AtomicIntent]) -> str:
        """Create a prompt template for a set of intents"""
        base_template = self._get_intent_description(intents)
        return f"Process the content with the following characteristics: {base_template}\n\nInput text:\n{{input_text}}"
    
    def process_input(self, task: str, dataset_key: str, input_text: str, user_id: str) -> str:
        """Process input text using specified intents for a specific user and task"""
        if task not in self.intents:
            raise ValueError(f"No intents defined for task: {task}")
            
        if dataset_key not in self.intents[task]:
            raise ValueError(f"No intents defined for dataset {dataset_key} in task {task}")
            
        if user_id not in self.intents[task][dataset_key]:
            raise ValueError(f"No intents defined for user {user_id} in dataset {dataset_key} for task {task}")
            
        intent_config = self.intents[task][dataset_key][user_id]
            
        # Construct the prompt with global guidelines and intent-specific template
        prompt = f"{GLOBAL_GUIDELINES[task][dataset_key]}\n\n"
        prompt += intent_config.prompt_template.format(input_text=input_text)
        
        # Call the model using the provided caller function
        return self.model_caller(prompt, input_text)

# Example usage:
def create_default_intents():
    """Create default intent configurations"""
    return [
        IntentConfig(
            intents={AtomicIntent.SUMMARIZE},
            description="Summarize the input text concisely",
            prompt_template="Please provide a concise summary of the following text:\n{input_text}"
        ),
        IntentConfig(
            intents={AtomicIntent.ANALYZE},
            description="Analyze the input text for key insights",
            prompt_template="Please analyze the following text and provide key insights:\n{input_text}"
        ),
        IntentConfig(
            intents={AtomicIntent.COMPARE},
            description="Compare the input text with other relevant information",
            prompt_template="Please compare the following text with relevant context:\n{input_text}"
        ),
        IntentConfig(
            intents={AtomicIntent.EXPLAIN},
            description="Explain the input text in detail",
            prompt_template="Please provide a detailed explanation of the following text:\n{input_text}"
        ),
        IntentConfig(
            intents={AtomicIntent.RECOMMEND},
            description="Provide recommendations based on the input text",
            prompt_template="Based on the following text, please provide recommendations:\n{input_text}"
        )
    ] 