import pyphen
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def count_syllables(text, dic):
    words = text.strip().split()
    syllable_count = 0
    for word in words:
        hyphenated = dic.inserted(word)
        syllable_count += hyphenated.count('-') + 1
    return syllable_count

def is_haiku(text):
    dic = pyphen.Pyphen(lang='en')
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if len(lines) != 3:
        return False, "Not exactly 3 lines"
    expected_syllables = [5, 7, 5]
    actual_syllables = [count_syllables(line, dic) for line in lines]
    if actual_syllables == expected_syllables:
        return True, "Valid haiku"
    else:
        return False, f"Syllable pattern mismatch: {actual_syllables} (expected [5, 7, 5])"

def reward_haiku(text:str) -> int:
    return int(is_haiku(text)[0])

def reward_similarity(summary:str, haiku:str):
    embedding_1 = model.encode(haiku, convert_to_tensor=True)
    embedding_2 = model.encode(summary, convert_to_tensor=True)
    value = util.pytorch_cos_sim(embedding_1, embedding_2)
    return value - 0.5

def compute_train_rewards(prompts, completions, **kwargs):
    question = prompts[0][0]["content"]
    assignment = prompts[0][0]["content"].split(":")[1].strip()
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        r.strip()
        for r in responses
    ]

    scores = []
    print('*'*20, f"Question:\n{question}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for response in extracted_responses:
        scores.append(reward_haiku(response) + reward_similarity(assignment, response))
        continue
    return scores