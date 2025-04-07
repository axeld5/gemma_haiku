import json
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from rewards import compute_train_rewards
from transformers import AutoModelForCausalLM

with open("train_dataset/rl_data.json", "r") as f:
    data = json.load(f)
rows = []
for example in data:
    for turn in example["conversations"]:
        if turn["role"] == "user":
            question = turn["content"].strip()
            prompt = question
            rows.append({"question": question, "prompt": turn})
            break
dataset = Dataset.from_list(rows)

max_seq_length = 2048
lora_rank = 32  # Larger rank = smarter, but slower


model_id = "google/gemma-3-1b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

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
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 50,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

def reward_bad():
    return 0

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [
        reward_bad
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_pretrained("gemma-3-rl")  # Local saving