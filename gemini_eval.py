from rewards import reward_similarity, reward_haiku
import google.generativeai as genai
import tqdm
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

def eval_model(api_key, prompts):
    """
    Evaluate Gemini 2.0 Flash model on the given prompts.
    
    Args:
        api_key (str): Google API key for accessing Gemini
        prompts (list): List of prompt dictionaries
        
    Returns:
        dict: Dictionary containing average haiku score, similarity score, and total score
    """
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the Gemini 2.0 Flash model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    n = len(prompts)
    haiku_val = 0
    similarity_val = 0
    val = 0
    
    for prompt in tqdm.tqdm(prompts):
        # Extract the prompt content
        prompt_content = prompt["content"]
        
        # Generate response using Gemini
        response = model.generate_content(prompt_content)
        generated_text = response.text.strip()
        
        # Extract the assignment from the prompt
        assignment = prompt_content.split(":")[1].strip()
        
        # Calculate rewards
        haiku_val += reward_haiku(generated_text)/n
        similarity_val += (reward_similarity(assignment, generated_text)/n).item()
        val = (haiku_val + similarity_val)
    
    return {
        "is_haiku_avg": haiku_val, 
        "similarity_avg": similarity_val, 
        "total_avg": val
    }

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("json", data_files="eval_dataset/eval_data.json")["train"]
    prompts = [elem["conversations"][0] for elem in dataset]
    
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Evaluate Gemini 2.0 Flash
    print(f"Average reward for Gemini 2.0 Flash: {eval_model(api_key, prompts)}") 