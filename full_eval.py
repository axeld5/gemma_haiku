from rewards import reward_similarity, reward_haiku
from unsloth import FastModel
import tqdm
from datasets import load_dataset
import json
import os
from collections import defaultdict
import csv
import datetime

def load_training_answers(file_path="train_dataset/sft_data.json"):
    """
    Load all answers from the training dataset to check for duplicates.
    
    Args:
        file_path (str): Path to the training dataset JSON file
        
    Returns:
        dict: Dictionary mapping answers to their frequency in the training data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract all answers from the training data
        answers = defaultdict(int)
        for item in data:
            for conversation in item["conversations"]:
                if conversation["role"] == "assistant":
                    answer = conversation["content"].strip()
                    answers[answer] += 1
        
        return answers
    except Exception as e:
        print(f"Error loading training answers: {e}")
        return {}

def load_model(model_path):
    """
    Load a model from the specified path.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        tuple: (model, tokenizer) if successful, (None, None) if model not available
    """
    try:
        print(f"Loading model from: {model_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        print(f"Model {model_path} is not available. Skipping.")
        return None, None

def eval_model(model_path, prompts, print_interval=10, check_duplicates=True):
    """
    Evaluate a model on the given prompts with additional features.
    
    Args:
        model_path (str): Path to the model to evaluate
        prompts (list): List of prompt dictionaries
        print_interval (int): Print query and answer every N inferences
        check_duplicates (bool): Whether to check for duplicates in training data
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load training answers if checking for duplicates
    training_answers = {}
    if check_duplicates:
        training_answers = load_training_answers()
        print(f"Loaded {len(training_answers)} unique answers from training data")
    
    # Load the model
    model, tokenizer = load_model(model_path)
    
    # Skip evaluation if model is not available
    if model is None or tokenizer is None:
        return {
            "model_path": model_path,
            "is_haiku_avg": 0,
            "similarity_avg": 0,
            "total_avg": 0,
            "duplicates": 0,
            "duplicate_percentage": 0,
            "total_samples": 0,
            "error": "Model not available"
        }
    
    n = len(prompts)
    haiku_val = 0
    similarity_val = 0
    val = 0
    duplicates = 0
    
    # Print header for detailed output
    print("\n" + "="*80)
    print(f"Evaluating model: {model_path}")
    print("="*80)
    
    for i, prompt in enumerate(tqdm.tqdm(prompts)):
        messages = [prompt]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
        )
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=50,
            temperature=1.0, top_p=0.95, top_k=64,
        )
        generated_text = tokenizer.batch_decode(outputs)[0].split("model\n")[1].split("<end_of_turn>")[0].strip()
        assignment = prompt["content"].split(":")[1].strip()
        
        # Calculate rewards
        haiku_score = reward_haiku(generated_text)/n
        similarity_score = (reward_similarity(assignment, generated_text)/n).item()
        haiku_val += haiku_score
        similarity_val += similarity_score
        val = (haiku_val + similarity_val)
        
        # Check if the answer is in the training data
        is_duplicate = False
        if check_duplicates and generated_text in training_answers:
            duplicates += 1
            is_duplicate = True
        
        # Print query and answer every print_interval inferences
        if (i + 1) % print_interval == 0:
            print(f"\n--- Inference {i+1}/{n} ---")
            print(f"Query: {assignment}")
            print(f"Answer: {generated_text}")
            print(f"Haiku Score: {haiku_score*n:.2f}")
            print(f"Similarity Score: {similarity_score*n:.2f}")
            if is_duplicate:
                print(f"⚠️ DUPLICATE: This answer appears {training_answers[generated_text]} times in training data")
            print("-" * 80)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Total samples: {n}")
    print(f"Average haiku score: {haiku_val:.4f}")
    print(f"Average similarity score: {similarity_val:.4f}")
    print(f"Total average score: {val:.4f}")
    if check_duplicates:
        print(f"Duplicate answers: {duplicates} ({duplicates/n*100:.2f}%)")
    print("="*80)
    
    return {
        "model_path": model_path,
        "is_haiku_avg": haiku_val,
        "similarity_avg": similarity_val,
        "total_avg": val,
        "duplicates": duplicates,
        "duplicate_percentage": duplicates/n if n > 0 else 0,
        "total_samples": n
    }

def save_results_to_csv(results, filename=None):
    """
    Save evaluation results to a CSV file.
    
    Args:
        results (list): List of dictionaries containing evaluation results
        filename (str, optional): Name of the CSV file. If None, a timestamped name will be used.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.csv"
    
    fieldnames = ["model_path", "is_haiku_avg", "similarity_avg", "total_avg", 
                 "duplicates", "duplicate_percentage", "total_samples", "error"]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Ensure all fields are present
            for field in fieldnames:
                if field not in result:
                    result[field] = ""
            writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Load the evaluation dataset
    dataset = load_dataset("json", data_files="eval_dataset/eval_data.json")["train"]
    prompts = [elem["conversations"][0] for elem in dataset]
    
    # List of models to evaluate (from README)
    models = [
        "unsloth/gemma-3-1b-it",           # Base model
        "gemma-3-1b-haiku",                 # Finetuned model
        "gemma-3-1b-sftrl-haiku",           # Finetuned+RL model
        "gemma-3-1b-sftrl-haiku-sparse",    # Finetuned+RL model (sparse)
        "gemma-3-haiku-rl-sparse",          # Sole RL model
        "gemma-3-1b-fullrun"                # Finetuned+RL model with sparse then continuous reward
    ]
    
    # Get models from command line if provided
    import sys
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    
    # Run evaluation for each model
    all_results = []
    for model_path in models:
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*80}")
        
        results = eval_model(model_path, prompts)
        all_results.append(results)
        
        # Print final results for this model
        print("\nFINAL RESULTS:")
        print(f"Model: {model_path}")
        if "error" in results and results["error"]:
            print(f"Error: {results['error']}")
        else:
            print(f"Haiku Score: {results['is_haiku_avg']:.4f}")
            print(f"Similarity Score: {results['similarity_avg']:.4f}")
            print(f"Total Score: {results['total_avg']:.4f}")
            print(f"Duplicate Answers: {results['duplicates']} ({results['duplicate_percentage']*100:.2f}%)")
    
    # Save all results to a CSV file
    save_results_to_csv(all_results)
    
    # Print a summary table of all results
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    print(f"{'Model':<40} {'Haiku':<10} {'Similarity':<10} {'Total':<10} {'Duplicates':<10} {'Status':<10}")
    print("-"*80)
    for result in all_results:
        status = "Error" if "error" in result and result["error"] else "Success"
        print(f"{result['model_path']:<40} {result['is_haiku_avg']:.4f}    {result['similarity_avg']:.4f}    {result['total_avg']:.4f}    {result['duplicate_percentage']*100:.2f}%    {status:<10}")
    print("="*80) 