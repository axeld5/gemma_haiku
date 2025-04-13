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

For more detailed evaluation with sample outputs and duplicate detection:

```bash
python full_eval.py
```

### Gemini Model Evaluation

To evaluate the Gemini 2.0 Flash model:

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

### Gemini 2.0 Flash

| Model | Haiku Score | Similarity Score | Total Score | Train Overlap |
|-------|-------------|------------------|-------------|------------|
| Gemini 2.0 Flash | 0.2044 | -0.0919 | 0.1125 | 0.00% |

## Model Performance Analysis

### ⚠️ Important Note on Results

**Please take these results with a grain of salt**: The Gemini Flash model appears to generate haikus very well on the 57 examples tested, but the pyphen function used for syllable counting is flawed, which means it gets penalized. Its score should be much higher otherwise.

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
