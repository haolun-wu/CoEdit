from src.task.abstract_task import Task
from src.task.dataset_helpers import load_data
from src.task.cost import get_cost_func
from src.correction import Correction
from global_user_intents import AtomicIntent, GLOBAL_GUIDELINES

import numpy as np
from typing import Tuple, Iterable, Optional, List, Set

class EmailWriting(Task):
    def __init__(self, task_config):
        self._data = EmailWriting._get_dataset(
            task_config.datasets or ['slf5k', 'ccby', 'ampere', 'paper_tweet'],
            task_config.num_train_ex, task_config.seed, )
        self._cost = get_cost_func(task_config.cost) 

    @staticmethod
    def _get_dataset(datasets, num_train_ex, seed):
        from itertools import chain, islice
        result = []
        num_doc_types = len(datasets)
        # Handle case when num_train_ex is -1 (use all examples)
        if num_train_ex == -1:
            num_ex_per_doc_type = -1
        else:
            # Ensure we don't divide by zero and handle edge cases
            num_ex_per_doc_type = max(1, int(num_train_ex / max(1, num_doc_types)))
        
        for dataset in datasets:
            result.append(list(load_data(dataset=dataset,
                                    num_ex=-1,
                                    split='train')))
        rng = np.random.default_rng(seed=seed)
        for r in result:
            rng.shuffle(r)
        # Use -1 for islice when we want all examples
        result = list(chain.from_iterable(map(lambda r: islice(r, num_ex_per_doc_type if num_ex_per_doc_type > 0 else None), result)))
        rng.shuffle(result)
        return result

    def next(self) -> Iterable[Tuple[str, Set[AtomicIntent], str]]:
        """
        Iterating over tuples (next_message, user_preference, dataset_type)
        """
        for d in self._data:
            yield d.article, d.user_pref, d.doc_type

    def get_edit_prompts(self, input: str, output: str, preference: Set[AtomicIntent], dataset_type: str) -> Tuple[str, str]:
        preference_str = ", ".join(intent.value for intent in preference)
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        
        resolution_prompt = "\n".join([
            f"Notes:\n{input}",
            f"Email:\n{output}",
            f"Dataset guidelines: {dataset_guideline}",
            f"User preferences: {preference_str}",
            f"Is the above email based on the above notes good for a user who wants the following style: {preference_str}? Please answer yes or no."])
        
        edit_prompt = "\n".join([
            f"Email:\n{output}",
            f"Dataset guidelines: {dataset_guideline}",
            f"User preferences: {preference_str}",
            f"Please revise the above email to meet both the dataset guidelines and user preferences."])
        return resolution_prompt, edit_prompt
    
    def get_task_prompt(self, input: str, preference: Optional[Set[AtomicIntent]] = None, dataset_type: str = None) -> str:
        if preference is None:
            return "\n".join([
                f"Notes:\n{input}",
                f"Please write a short email based on your above notes."])
        
        preference_str = ", ".join(intent.value for intent in preference)
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        
        return "\n".join([
            f"Notes:\n{input}",
            f"Dataset guidelines: {dataset_guideline}",
            f"User preferences: {preference_str}",
            f"Please write a short email based on the above notes that follows both the dataset guidelines and user preferences."])

    def get_task_prompt_icl(self, input: str, corrections: List[Correction], dataset_type: str) -> str:
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original email:\n{correction.original.text}\n'
            prompt = prompt + f'Revised email:\n{correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Notes:\n{input}",
            f"Dataset guidelines: {dataset_guideline}",
            f"Based on the edits and revision by this user on the original email in the above examples, please write an email based on the above notes following the dataset guidelines."])
        return prompt
    
    def get_task_prompt_icl_pref(self, input: str, preferences: List[Set[AtomicIntent]], dataset_type: str) -> str:
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        prompt = 'List of user preferences successfully being used to generate emails of a similar kind:\n'
        for preference in preferences:
            preference_str = ", ".join(intent.value for intent in preference)
            prompt = prompt + f'- {preference_str}\n'
        prompt += "\n".join([
            f"Notes:\n{input}",
            f"Dataset guidelines: {dataset_guideline}",
            f"Using the qualities most represented in the above list of preferences and following the dataset guidelines, please write an email based on the above notes."])
        return prompt

    def get_preference_inference_prompt(self, corrections: List[Correction], dataset_type: str) -> str:
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        prompt = ''
        for correction in corrections:
            prompt = prompt + f'Original email:\n{correction.original.text}\n'
            prompt = prompt + f'Revised email:\n{correction.edited.text}\n\n'
        prompt += "\n".join([
            f"Dataset guidelines: {dataset_guideline}",
            f"Based on the edits and revision by this user on the original email in the above examples, what do you find about this user's generic preference in terms of writing style and formatting?",
            f"Please answer in a short phrase and only recommend those preferences that are widely used."])
        return prompt 

    def get_majority_preference_prompt(self, preferences: List[Set[AtomicIntent]], dataset_type: str) -> str:
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        prompt = 'List of user preferences successfully being used to generate emails of a similar kind:\n'
        for preference in preferences:
            preference_str = ", ".join(intent.value for intent in preference)
            prompt += f'- {preference_str}\n'
        prompt += f"Dataset guidelines: {dataset_guideline}\n"
        prompt += "Based on the above examples and dataset guidelines, please come up with short phrase with the most represented writing preferences of this user."
        return prompt

    def get_base_prompt(self, input: str, dataset_type: str = None) -> str:
        """Get prompt for the base model (Phi) that only uses dataset guidelines.
        
        Args:
            input: The input notes
            dataset_type: The type of dataset
            
        Returns:
            A prompt string for the base model
        """
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        
        return "\n".join([
            f"Notes:\n{input}",
            f"Dataset guidelines: {dataset_guideline}",
            f"Please write a short email based on the above notes following the dataset guidelines."])

    def get_edit_prompt(self, input: str, base_output: str, preference: Set[AtomicIntent], dataset_type: str) -> str:
        """Get prompt for the user simulation model (GPT) that edits the base output.
        
        Args:
            input: The input notes
            base_output: The output from the base model
            preference: The user's preferences
            dataset_type: The type of dataset
            
        Returns:
            A prompt string for the user simulation model
        """
        preference_str = ", ".join(intent.value for intent in preference)
        dataset_guideline = GLOBAL_GUIDELINES.get('email_writing', {}).get(dataset_type, "")
        
        return "\n".join([
            f"Notes:\n{input}",
            f"Base email:\n{base_output}",
            f"Dataset guidelines: {dataset_guideline}",
            f"User preferences: {preference_str}",
            f"Please revise the above base email to better match both the dataset guidelines and user preferences. Focus on making the email more aligned with the user's preferred style while maintaining the key information."])
