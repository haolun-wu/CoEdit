from src.task.intent_handler import IntentHandler
from src.task.dataset_helpers import load_data
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from src.utils.call_llms import create_model_caller
from global_user_intents import GLOBAL_GUIDELINES

def save_results(results: dict, output_dir: str = "synthesized", num_samples: int = 1):
    """Save results to JSON files"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each task's results in a separate file
    for task, task_results in results.items():
        # Group results by dataset
        dataset_results = {}
        for result in task_results:
            dataset = result["dataset"]
            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append(result)
        
        # Save each dataset's results in a separate file
        for dataset, dataset_data in dataset_results.items():
            filename = f"{output_dir}/{task}_{dataset}_samples{num_samples}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset_data, f, indent=2, ensure_ascii=False)
            print(f"Saved results to {filename}")

def main(test_mode: bool = False, test_samples: int = 1, model_name: str = "gpt-4o-mini"):
    # Create model caller and initialize intent handler
    model_caller = create_model_caller(model_name)
    handler = IntentHandler(model_caller)
    
    # Define tasks and their datasets
    tasks = {
        "summarization": ['cnn_dailymail'],
        # "email_writing": ['slf5k']
    }
    
    # Store results for each task
    results = {task: [] for task in tasks.keys()}
    
    # Process each task and its datasets
    for task, datasets in tasks.items():
        print(f"\nProcessing task: {task}")
        
        for dataset_name in datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            
            # Load dataset with multiple users
            num_examples = test_samples if test_mode else 100
            dataset = load_data(dataset_name, num_ex=num_examples, num_users=5)  # Use all 5 users
            
            # Get dataset guidelines
            dataset_guideline = GLOBAL_GUIDELINES.get(task, {}).get(dataset_name, "")
            
            # Process examples for each user
            for user_id in dataset.get_unique_users():
                print(f"\nProcessing examples for {user_id}:")
                user_examples = dataset.get_examples_by_user(user_id)
                
                for example in user_examples:
                    print(f"\nArticle ID: {example.id}")
                    print(f"Article preview: {example.article[:100]}...")
                    print(f"User preference: {example.user_pref}")
                    
                    try:
                        # Get the prompt before processing
                        prompt = handler._construct_prompt(task, example.article, user_id, dataset_name)
                        result = handler.process_input(task, dataset_name, example.article, user_id)
                        print(f"\n{'='*50}\nTask: {task}\nDataset: {dataset_name}\nUser: {user_id}\nPrompt:\n{prompt[:100]}...{prompt[-100:]}\n{'='*50}\n")
                        print(f"\nResponse:\n{result[:100]}...{result[-100:]}\n{'='*50}\n")
                        
                        # Convert user preferences to a single string
                        user_pref_str = ", ".join(intent.value for intent in example.user_pref) if example.user_pref else ""
                        
                        # Store the result with the prompt
                        results[task].append({
                            "dataset": dataset_name,
                            "user_id": user_id,
                            "article_id": example.id,
                            "article_preview": example.article[:100],
                            "dataset_guideline": dataset_guideline,  # Added dataset guidelines
                            "user_preference": user_pref_str,  # Now using a single string
                            "prompt": prompt,  # Save the exact prompt used
                            "result": result,
                            "model": model_name,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"Error processing: {str(e)}")
    
    # Save results to JSON files
    save_results(results, num_samples=test_samples if test_mode else 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets with user intents')
    parser.add_argument('--test-mode', type=bool, default=True, help='Run in test mode with limited samples')
    parser.add_argument('--test-samples', type=int, default=1, help='Number of samples to use in test mode')
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model name to use')
    parser.add_argument('--output-dir', type=str, default="synthesized", help='Directory to save results')
    args = parser.parse_args()
    
    main(test_mode=args.test_mode, test_samples=args.test_samples, model_name=args.model) 