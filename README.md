# Gemma Haiku Project

This project fine-tunes Gemma models to generate haikus using different training approaches:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning (RL) with sparse rewards
- RL with regular rewards
- Combined SFT + RL approaches

## Models

The project trains and evaluates the following models:
1. `gemma-3-1b-haiku` - Pure SFT model
2. `gemma-3-haiku-rl-sparse` - Pure RL with sparse rewards
3. `gemma-3-1b-sftrl-haiku` - SFT + RL with regular rewards
4. `gemma-3-1b-sftrl-haiku-sparse` - SFT + RL with sparse rewards
5. `gemma-3-1b-fullrun` - SFT + RL with sparse rewards + RL with regular rewards

## Using Docker

The easiest way to run this project is using Docker:

### Prerequisites

- Docker installed on your system
- NVIDIA GPU with CUDA support (for training)

### Building the Docker Image

```bash
docker build -t gemma-haiku .
```

### Running the Container

```bash
docker run --gpus all -v $(pwd)/models:/app/models -v $(pwd)/results:/app/eval_results gemma-haiku
```

This will:
1. Train all models using the `train_models.py` script
2. Evaluate all models using the `full_eval.py` script
3. Save the evaluation results to the `results` directory

### Customizing Training

You can customize the training by modifying the Dockerfile or by running the container with different commands:

```bash
# Train only specific models
docker run --gpus all -v $(pwd)/models:/app/models gemma-haiku python train_models.py --train_pure_sft --train_pure_rl_sparse

# Evaluate only specific models
docker run --gpus all -v $(pwd)/models:/app/models -v $(pwd)/results:/app/eval_results gemma-haiku python full_eval.py gemma-3-1b-haiku gemma-3-haiku-rl-sparse
```

## Manual Setup

If you prefer to run the project without Docker:

1. Install Python 3.12
2. Install the requirements: `pip install -r requirements.txt`
3. Run the training script: `python train_models.py`
4. Run the evaluation script: `python full_eval.py`

## Project Structure

- `train_models.py` - Script to train all models
- `gemma_sft.py` - SFT training implementation
- `gemma_rl.py` - RL training implementation
- `full_eval.py` - Evaluation script
- `rewards.py` - Reward functions for RL training
- `train_dataset/` - Directory for training data
- `eval_dataset/` - Directory for evaluation data
- `eval_results/` - Directory for evaluation results

## Overview

The project aims to train language models to generate haikus that:
1. Follow the traditional 5-7-5 syllable pattern
2. Maintain semantic similarity to the given prompt
3. Produce high-quality, coherent haikus

## Repository Structure

- `gemma_sft.py` - Supervised Fine-Tuning implementation for Gemma model
- `gemma_rl.py` - Reinforcement Learning implementation for Gemma model
- `gemini_eval.py` - Evaluation script for Gemini 2.0 Flash model
- `full_eval.py` - Enhanced evaluation script with detailed output
- `rewards.py` - Reward functions for haiku generation evaluation
- `prompts.py` - Prompt templates for model interactions
- `train_dataset/` - Training data for SFT and RL approaches
- `eval_dataset/` - Evaluation data for model assessment
- `haiku_dataset/` - Dataset of haikus for training

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
```

## Training Approaches

### 1. Supervised Fine-Tuning (SFT)

The SFT approach uses the `gemma_sft.py` script to fine-tune the Gemma model on haiku generation tasks:

```bash
python gemma_sft.py --model_name "unsloth/gemma-3-1b-it" --max_steps 200 --save_path "gemma-3-1b-haiku"
```

This will:
- Load the base Gemma 3 1B model
- Apply LoRA fine-tuning
- Train on the SFT dataset
- Save the model as "gemma-3-1b-haiku"

### 2. Reinforcement Learning (RL)

The RL approach uses the `gemma_rl.py` script to train the model using GRPO (Generative Reinforcement Policy Optimization):

```bash
python gemma_rl.py --model_name "unsloth/gemma-3-1b-it" --max_steps 200 --save_path "gemma-3-haiku-rl-sparse" --is_reward_sparse True
```

This will:
- Load the base Gemma model
- Apply LoRA fine-tuning
- Train using the RL dataset with custom reward functions
- Save the model as "gemma-3-haiku-rl-sparse"

### 3. Combined SFT+RL Approach

The combined approach uses the `train_models.py` script to first fine-tune with SFT and then further improve with RL:

```bash
python train_models.py --train_sft_rl_sparse_regular
```

This will:
- Load a pre-fine-tuned model
- Apply additional RL training
- Save the final model as "gemma-3-1b-fullrun"

## Evaluation

### Model Evaluation

To evaluate all models:

```bash
python full_eval.py
```

For evaluating specific models:

```bash
python full_eval.py gemma-3-1b-haiku gemma-3-haiku-rl-sparse
```

To perform evaluation specifically on Google Gemini-2.0-Flash:
```bash
python gemini_eval.py
```

## Evaluation Results

### Gemma Models

| Model | Haiku Score | Similarity Score | Total Score | Train Overlap |
|-------|-------------|------------------|-------------|------------|
| unsloth/gemma-3-1b-it | 0.0372 | -0.0998 | -0.0627 | 0.00% |
| gemma-3-1b-haiku | 0.1351 | 0.1101 | 0.2453 | 0.00% |
| gemma-3-1b-sftrl-haiku | 0.0878 | 0.3708 | 0.4587 | 0.00% |
| gemma-3-1b-sftrl-haiku-sparse | 0.1858 | -0.0880 | 0.0978 | 0.00% |
| gemma-3-haiku-rl-sparse | 0.1537 | -0.1206 | 0.0331 | 0.00% |
| gemma-3-1b-fullrun | 0.2348 | 0.0588 | 0.2936 | 0.00% |

## Model Performance Analysis

### Detailed Model Analysis

#### Base Model (unsloth/gemma-3-1b-it)
- **Low Score Explanation**: The base model's low score is primarily due to the added messages like "sure, I will generate a Haiku" that it includes in its responses, which don't follow the haiku format.

#### Finetuned Model (gemma-3-1b-haiku)
- **Performance Analysis**: The finetuned model achieves a moderate score but still struggles with the syllable requirements. It generally gets the "three lines" requirement correct but fails to get the syllable count right, resulting in haikus that are structurally incorrect.

#### Finetuned+RL Model (gemma-3-1b-sftrl-haiku)
- **Interesting Pattern**: This model shows an interesting pattern where it "hacks" the similarity reward. It first generates the words from the prompt and then fits them into a haiku-like structure. This explains both the moderate haiku score and the high similarity score, but it's not truly generating creative haikus.

#### Finetuned+RL Model with Sparse Rewards (gemma-3-1b-sftrl-haiku-sparse)
- **Partial Success**: This model shows signs of improvement in haiku generation but fails at following the instruction properly. It generates haiku-like outputs but doesn't always adhere to the prompt requirements.

#### Sole RL Model with Sparse Rewards (gemma-3-haiku-rl-sparse)
- **Incomplete Learning**: This model sometimes generates too much text instead of just a haiku. A longer training runtime would be required for it to completely understand the 3-line penalty and generate proper haikus consistently.

#### Combined Approach (gemma-3-1b-fullrun)
- **Best Performance**: This model represents the best of both worlds - it follows the 3-line requirement and instruction following from SFT, while also achieving a good balance of correct syllable counts from the RL component. It achieves the highest haiku score among all models.

## Reward Functions

The `rewards.py` file contains several reward functions:

- `reward_haiku()` - Checks if the generated text follows the 5-7-5 syllable pattern
- `reward_similarity()` - Measures semantic similarity between the prompt and generated haiku
- `compute_train_rewards()` - Combines rewards for RL training
- `compute_train_rewards_sparse()` - Sparse version of the combined rewards

## Datasets

- `train_dataset/sft_data.json` - Data for supervised fine-tuning
- `train_dataset/rl_data.json` - Data for reinforcement learning
- `eval_dataset/eval_data.json` - Data for model evaluation
- `train_dataset/train_haiku.csv` - Additional haiku training data

## License

[Add your license information here]

## Acknowledgements

- Unsloth for the FastModel implementation
- Google for the Gemma and Gemini models
- Hugging Face for the Transformers library and TRL framework
