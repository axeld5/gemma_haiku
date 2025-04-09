from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import json
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer
from rewards import compute_train_rewards

with open("train_dataset/rl_data.json", "r") as f:
    rl_data = json.load(f)
with open("train_dataset/sft_data.json", "r") as f:
    sft_data = json.load(f)
data = rl_data
rows = []
for example in data:
    for turn in example["conversations"]:
        if turn["role"] == "user":
            question = turn["content"].strip()
            prompt = question
            rows.append({"question": question, "prompt": [turn]})
            break

dataset = Dataset.from_list(rows)

max_seq_length = 512
model, tokenizer = FastModel.from_pretrained(
    model_name = "gemma-3-1b-haiku",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 10, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        compute_train_rewards
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_pretrained("gemma-3-1b-sftrl-haiku")  # Local saving
tokenizer.save_pretrained("gemma-3-1b-sftrl-haiku")