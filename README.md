# Gemma Haiku Generator

This repository contains code for fine-tuning and evaluating language models to generate haikus based on given prompts. The project includes implementations for both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) approaches, as well as evaluation scripts for both Gemma and Gemini models.

## Overview

The project aims to train language models to generate haikus that:
1. Follow the traditional 5-7-5 syllable pattern
2. Maintain semantic similarity to the given prompt
3. Produce high-quality, coherent haikus

## Repository Structure

- `gemma_sft.py` - Supervised Fine-Tuning implementation for Gemma model
- `gemma_rl.py` - Reinforcement Learning implementation for Gemma model
- `gemma_sft_rl.py` - Combined SFT+RL approach for Gemma model
- `gemma_eval.py` - Evaluation script for Gemma models
- `gemini_eval.py` - Evaluation script for Gemini 2.0 Flash model
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
python gemma_sft.py
```

This will:
- Load the base Gemma 3 1B model
- Apply LoRA fine-tuning
- Train on the SFT dataset
- Save the model as "gemma-3-1b-haiku"

### 2. Reinforcement Learning (RL)

The RL approach uses the `gemma_rl.py` script to train the model using GRPO (Generative Reinforcement Policy Optimization):

```bash
python gemma_rl.py
```

This will:
- Load the base Gemma model
- Apply LoRA fine-tuning
- Train using the RL dataset with custom reward functions
- Save the model as "gemma-3-haiku-rl-sparse"

### 3. Combined SFT+RL Approach

The combined approach uses the `gemma_sft_rl.py` script to first fine-tune with SFT and then further improve with RL:

```bash
python gemma_sft_rl.py
```

This will:
- Load a pre-fine-tuned model
- Apply additional RL training
- Save the final model as "gemma-3-1b-fullrun"

## Evaluation

### Gemma Model Evaluation

To evaluate Gemma models:

```bash
python gemma_eval.py
```

This will evaluate the model's performance on the evaluation dataset and report:
- Haiku validity score
- Semantic similarity score
- Total combined score

### Gemini Model Evaluation

To evaluate the Gemini 2.0 Flash model:

```bash
python gemini_eval.py
```

This will evaluate Gemini's performance using the same metrics as the Gemma models.

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
