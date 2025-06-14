from src.task.intent_handler import IntentHandler
from src.task.dataset_helpers import load_data, print_dataset_stats
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from src.utils.call_llm_helpers import create_model_caller
from global_user_intents import GLOBAL_GUIDELINES
from tqdm import tqdm


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


def main(test_mode: bool = False, test_samples: int = 1, base_model: str = "microsoft/Phi-4-mini-instruct", edit_model: str = "gpt-4o-mini"):
   # Create model callers for both stages
   base_model_caller = create_model_caller(base_model)  # First stage: Base model
   edit_model_caller = create_model_caller(edit_model)  # Second stage: Edit model
  
   # Initialize intent handler with the edit model
   handler = IntentHandler(edit_model_caller)
  
   # Define tasks and their datasets
   tasks = {
       "summarization": ['cnn_dailymail'],
       # "email_writing": ['slf5k']
   }
  
   # Store results for each task
   results = {task: [] for task in tasks.keys()}
  
   # Calculate total examples to process
   total_examples = sum(len(datasets) * test_samples for datasets in tasks.values())
   pbar = tqdm(total=total_examples, desc="Processing examples")
  
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
            #    print(f"\nProcessing examples for {user_id}:")
               user_examples = dataset.get_examples_by_user(user_id)
              
               for example in user_examples:
                #    print(f"\nArticle ID: {example.id}")
                #    print(f"Article preview: {example.article[:100]}...")
                #    print(f"User preference: {example.user_pref}")
                  
                   try:
                       # Get task instance for prompt construction
                       task_instance = handler._get_task(task, dataset_name)
                      
                       # Stage 1: Generate base output using base model
                       base_prompt = task_instance.get_base_prompt(example.article, dataset_name)
                       base_output = base_model_caller(base_prompt)
                       if base_output.startswith(base_prompt): # Remove base_prompt from the base_output
                           base_output = base_output[len(base_prompt):].strip()
                    #    print(f"Base output:\n{base_output}")
                      
                       # Stage 2: Generate edited output using edit model
                       edit_prompt = task_instance.get_edit_prompt(
                           example.article,
                           base_output,
                           example.user_pref,
                           dataset_name
                       )
                       edit_output = edit_model_caller(edit_prompt)
                      
                    #    print(f"\n{'='*50}\nTask: {task}\nDataset: {dataset_name}\nUser: {user_id}")
                    #    print(f"Base prompt:\n{base_prompt[:100]}...{base_prompt[-100:]}")
                    #    print(f"Edit prompt:\n{edit_prompt[:100]}...{edit_prompt[-100:]}")
                    #    print(f"{'='*50}\n")
                    #    print(f"Base output:\n{base_output[:100]}...{base_output[-100:]}")
                    #    print(f"Edited output:\n{edit_output[:100]}...{edit_output[-100:]}\n{'='*50}\n")
                      
                       # Convert user preferences to a single string
                       user_pref_str = ", ".join(intent.value for intent in example.user_pref) if example.user_pref else ""
                      
                       # Store the results with both outputs
                       results[task].append({
                           "dataset": dataset_name,
                           "user_id": user_id,
                           "article_id": example.id,
                           "article_preview": f"{example.article[:100]}...{example.article[-100:]}",
                           "dataset_guideline": dataset_guideline,
                           "user_preference": user_pref_str,
                           "base_prompt": base_prompt,
                           "edit_prompt": edit_prompt,
                           "base_output": base_output,
                           "edit_output": edit_output,
                           "base_model": base_model,
                           "edit_model": edit_model,
                           "timestamp": datetime.now().isoformat()
                       })
                   except Exception as e:
                       print(f"Error processing: {str(e)}")
                   finally:
                       pbar.update(1)
  
   pbar.close()
   # Save results to JSON files
   save_results(results, num_samples=test_samples if test_mode else 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets with user intents')
    parser.add_argument('--test-mode', type=bool, default=True, help='Run in test mode with limited samples')
    parser.add_argument('--test-samples', type=int, default=2, help='Number of samples to use in test mode')
    parser.add_argument('--base-model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model name to use for base generation')
    parser.add_argument('--edit-model', type=str, default="gpt-4o-mini", help='Model name to use for editing')
    parser.add_argument('--output-dir', type=str, default="synthesized", help='Directory to save results')
    parser.add_argument('--show-stats', type=str, nargs='+', default=['cnn_dailymail'], help='Show statistics for one or more datasets (e.g., cnn_dailymail xsum slf5k)')
    args = parser.parse_args()

    if args.show_stats:
        for dataset in args.show_stats:
            print_dataset_stats(dataset)
    
    main(test_mode=args.test_mode, test_samples=args.test_samples, 
         base_model=args.base_model, edit_model=args.edit_model)