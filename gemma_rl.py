from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer
from rewards import compute_train_rewards

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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
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
    processing_class = tokenizer,
    reward_funcs = [
        reward_bad
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

model.save_pretrained("gemma-3-rl")  # Local saving
tokenizer.save_pretrained("gemma-3-rl")