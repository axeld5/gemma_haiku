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
    output_dir="gemma-3-1b-rl",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=None,
    logging_steps=10,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)

def reward_bad():
    return 0

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [
        compute_train_rewards
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_pretrained("gemma-3-rl")  # Local saving