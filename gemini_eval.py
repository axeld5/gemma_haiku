from rewards import reward_similarity, reward_haiku
import google.generativeai as genai
import tqdm
from datasets import load_dataset
import os
import json
from collections import defaultdict
import csv
import datetime
from dotenv import load_dotenv

load_dotenv()

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

def eval_model(api_key, prompts, print_interval=10, check_duplicates=True):
    """
    Evaluate Gemini 2.0 Flash model on the given prompts with detailed output.
    
    Args:
        api_key (str): Google API key for accessing Gemini
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
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the Gemini 2.0 Flash model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    n = len(prompts)
    haiku_val = 0
    similarity_val = 0
    val = 0
    duplicates = 0
    
    # Print header for detailed output
    print("\n" + "="*80)
    print(f"Evaluating model: Gemini 2.0 Flash")
    print("="*80)
    
    for i, prompt in enumerate(tqdm.tqdm(prompts)):
        # Extract the prompt content
        prompt_content = prompt["content"]
        
        # Generate response using Gemini
        response = model.generate_content(prompt_content)
        generated_text = response.text.strip()
        
        # Extract the assignment from the prompt
        assignment = prompt_content.split(":")[1].strip()
        
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
    print(f"Model: Gemini 2.0 Flash")
    print(f"Total samples: {n}")
    print(f"Average haiku score: {haiku_val:.4f}")
    print(f"Average similarity score: {similarity_val:.4f}")
    print(f"Total average score: {val:.4f}")
    if check_duplicates:
        print(f"Duplicate answers: {duplicates} ({duplicates/n*100:.2f}%)")
    print("="*80)
    
    return {
        "model_path": "Gemini 2.0 Flash",
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
        filename = f"gemini_evaluation_results_{timestamp}.csv"
    
    fieldnames = ["model_path", "is_haiku_avg", "similarity_avg", "total_avg", 
                 "duplicates", "duplicate_percentage", "total_samples"]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("json", data_files="eval_dataset/eval_data.json")["train"]
    prompts = [elem["conversations"][0] for elem in dataset]
    
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Run evaluation
    results = eval_model(api_key, prompts)
    
    # Save results to CSV
    save_results_to_csv([results])
    
    # Print final results
    print("\nFINAL RESULTS:")
    print(f"Model: Gemini 2.0 Flash")
    print(f"Haiku Score: {results['is_haiku_avg']:.4f}")
    print(f"Similarity Score: {results['similarity_avg']:.4f}")
    print(f"Total Score: {results['total_avg']:.4f}")
    print(f"Duplicate Answers: {results['duplicates']} ({results['duplicate_percentage']*100:.2f}%)") 