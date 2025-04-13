import json
import os
from huggingface_hub import HfApi, HfFolder, Repository
from dotenv import load_dotenv

load_dotenv()
# Define your Hugging Face model name and API token
model_name = "gemma-3-1b-fullrun"
api_token = os.environ.get("HF_TOKEN")  # Replace with your actual token

# Initialize the Hugging Face API
api = HfApi()

# Load and fuse JSON files
def fuse_json_files(file_paths):
    fused_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Determine the source type from the file path
            if 'sft_data.json' in file_path:
                source_type = 'sft'
            elif 'rl_data.json' in file_path:
                source_type = 'rl'
            elif 'eval_data.json' in file_path:
                source_type = 'eval'
            else:
                source_type = 'unknown'
            
            # Add source indicator to each data point
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item['source'] = source_type
                    fused_data.append(item)
            elif isinstance(data, dict):
                data['source'] = source_type
                fused_data.append(data)
    return fused_data

# Specify the paths to your JSON files
json_files = ["eval_dataset/eval_data.json", "train_dataset/sft_data.json", "train_dataset/rl_data.json"]  # Update with your actual paths
fused_json = fuse_json_files(json_files)

# Save the fused JSON to a new file
fused_json_path = "fused_data.json"
with open(fused_json_path, 'w') as f:
    json.dump(fused_json, f)

# Push the fused JSON file to the Hugging Face Hub
api.upload_file(
    path_or_fileobj=fused_json_path,
    path_in_repo=f"{model_name}/fused_data.json",
    repo_id=model_name,
    token=api_token
)

# Push the model to the Hugging Face Hub
repo = Repository(local_dir=model_name, clone_from=model_name, use_auth_token=api_token)

# Assuming your model files are in the current directory
repo.git_add(auto_lfs_track=True)
repo.git_commit("Add model files and fused JSON")
repo.git_push()