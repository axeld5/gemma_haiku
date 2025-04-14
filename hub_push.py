import json
import os
from huggingface_hub import HfApi, HfFolder, Repository
from dotenv import load_dotenv
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from huggingface_hub import login
import argparse
import os

model_name = "gemma-3-1b-fullrun"
login()
model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

model.push_to_hub("axeldarmouni/gemma-3-1b-haikuspec")