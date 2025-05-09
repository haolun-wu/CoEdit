from typing import List, Dict, Any, Set, Callable
import openai
import os
from dataclasses import dataclass
from enum import Enum
from src.task.abstract_task import Task
from src.task.dataset_helpers import OurInputExample
from src.task.cost import get_cost_func
from global_user_intents import USER_INTENTS, GLOBAL_GUIDELINES, AtomicIntent
from src.task.summarization import Summarization
from src.task.email_writing import EmailWriting

import numpy as np

@dataclass
class TaskConfig:
    """Configuration for tasks"""
    datasets: List[str] = None
    num_train_ex: int = -1
    seed: int = 42
    cost: str = "edit_distance"

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
        self._intents: Dict[str, Set[AtomicIntent]] = {}
        self._initialize_intents()
        self._tasks = {}  # Initialize empty, will create tasks as needed
        
    def _initialize_intents(self):
        """Initialize intents for each user."""
        for user_id, user_intent in USER_INTENTS.items():
            self._intents[user_id] = user_intent.intents
    
    def _get_intent_description(self, intents: Set[AtomicIntent]) -> str:
        """Convert a set of intents into a descriptive string."""
        return ", ".join(intent.value for intent in intents)
    
    def _get_task(self, task_name: str, dataset_name: str) -> Task:
        """Get or create a task instance with only the needed dataset."""
        if task_name not in self._tasks:
            # Create task config with only the dataset that is actually used
            task_config = TaskConfig(datasets=[dataset_name])
            if task_name == "summarization":
                self._tasks[task_name] = Summarization(task_config)
            elif task_name == "email_writing":
                self._tasks[task_name] = EmailWriting(task_config)
            else:
                raise ValueError(f"Unknown task: {task_name}. Supported tasks are: ['summarization', 'email_writing']")
        return self._tasks[task_name]
    
    def _construct_prompt(self, task: str, input_text: str, user_id: str, dataset_name: str) -> str:
        """Construct a prompt that combines user intents with global guidelines."""
        # Get user's intents
        user_intents = self._intents[user_id]
        
        # Get the appropriate task instance
        task_instance = self._get_task(task, dataset_name)
        
        # Use the task-specific prompt
        prompt = task_instance.get_task_prompt(input_text, user_intents, dataset_name)
        return prompt
    
    def process_input(self, task: str, dataset_name: str, input_text: str, user_id: str) -> str:
        """Process input with user intents and generate output."""
        prompt = self._construct_prompt(task, input_text, user_id, dataset_name)
        response = self.model_caller(prompt, input_text)
        return response

    def get_user_intents(self, user_id: str) -> Set[AtomicIntent]:
        """Get the intents for a specific user."""
        return self._intents.get(user_id, set())

    def get_all_intents(self) -> Dict[str, Set[AtomicIntent]]:
        """Get all user intents."""
        return self._intents.copy()