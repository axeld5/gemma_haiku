def summarize_haiku_prompt(haiku:str) -> str:
    prompt = f"""You are a model tasked to sum up very shortly what the haiku is about.
    <haiku> {haiku} </haiku>
    Output the summary and only the summary:"""
    return prompt