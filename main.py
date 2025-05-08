from src.task.intent_handler import IntentHandler
from src.task.dataset_helpers import load_data
import os
import argparse
import openai
import json
from datetime import datetime
from pathlib import Path

def create_model_caller(api_key: str, model_name: str = "gpt-4o-mini"):
    """Create a model caller function with the specified API key and model"""
    openai.api_key = api_key
    
    def model_caller(prompt: str, input_text: str) -> str:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content
    
    return model_caller

def save_results(results: dict, output_dir: str = "synthesized"):
    """Save results to JSON files"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each task's results in a separate file
    for task, task_results in results.items():
        filename = f"{output_dir}/{task}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(task_results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {filename}")

def main(test_mode: bool = False, test_samples: int = 2, model_name: str = "gpt-4o-mini"):
    # Load API key from secrets.txt
    with open('secrets.txt', 'r') as f:
        for line in f:
            if line.startswith('openai_key'):
                api_key = line.strip().split(',')[1]
                break
    
    # Create model caller and initialize intent handler
    model_caller = create_model_caller(api_key, model_name)
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
            num_examples = test_samples if test_mode else 5
            dataset = load_data(dataset_name, num_ex=num_examples, num_users=5)  # Use all 5 users
            
            # Process examples for each user
            for user_id in dataset.get_unique_users():
                print(f"\nProcessing examples for {user_id}:")
                user_examples = dataset.get_examples_by_user(user_id)
                
                for example in user_examples:
                    print(f"\nArticle ID: {example.id}")
                    print(f"Article preview: {example.article[:100]}...")
                    print(f"User preference: {example.user_pref}")
                    
                    try:
                        result = handler.process_input(task, dataset_name, example.article, user_id)
                        print(f"\nResult: {result[:200]}...")
                        
                        # Store the result
                        results[task].append({
                            "dataset": dataset_name,
                            "user_id": user_id,
                            "article_id": example.id,
                            "article_preview": example.article[:100],
                            "user_preference": example.user_pref,
                            "result": result,
                            "model": model_name,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"Error processing: {str(e)}")
    
    # Save results to JSON files
    save_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets with user intents')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited samples')
    parser.add_argument('--test-samples', type=int, default=2, help='Number of samples to use in test mode')
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model name to use')
    parser.add_argument('--output-dir', type=str, default="synthesized", help='Directory to save results')
    args = parser.parse_args()
    
    main(test_mode=args.test, test_samples=args.test_samples, model_name=args.model) 