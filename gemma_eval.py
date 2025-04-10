from rewards import reward_similarity, reward_haiku
from unsloth import FastModel
import tqdm
from datasets import load_dataset

def eval_model(model_path, prompts):
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 512,
        load_in_4bit = True,
    )
    n = len(prompts)
    haiku_val = 0
    similarity_val = 0
    val = 0
    for prompt in tqdm.tqdm(prompts):
        messages = [prompt]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
        )
        outputs = model.generate(
            **tokenizer([text], return_tensors = "pt").to("cuda"),
            max_new_tokens = 50,
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )
        generated_text = tokenizer.batch_decode(outputs)[0].split("model\n")[1].split("\n<end_of_turn>")[0].strip()
        assignment = prompt["content"].split(":")[1].strip()
        haiku_val += reward_haiku(generated_text)/n
        similarity_val += (reward_similarity(assignment, generated_text)/n).item()
        val = (haiku_val+similarity_val)
    return {"is_haiku_avg":haiku_val, "similarity_avg":similarity_val, "total_avg": val}

if __name__ == "__main__":
    dataset = load_dataset("json", data_files="eval_dataset/eval_data.json")["train"]
    prompts = [elem["conversations"][0] for elem in dataset]
    #print(f"Average reward for the base model: {eval_model('unsloth/gemma-3-1b-it', prompts)}")
    #print(f"Average reward for the finetuned model: {eval_model('gemma-3-1b-haiku', prompts)}")
    #print(f"Average reward for the finetuned+RL model: {eval_model('gemma-3-1b-sftrl-haiku', prompts)}")
    print(f"Average reward for the finetuned+RL model: {eval_model('gemma-3-1b-sftrl-haiku-sparse', prompts)}")
    print(f"Average reward for the sole RL model: {eval_model('gemma-3-haiku-rl-sparse', prompts)}")