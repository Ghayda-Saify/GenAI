def build_prompt(text: str) -> str:
    return f"""
    You are a helpful AI assistant.

    TASK:
    Summarize the following text in 3 bullet points.

    TEXT:
    {text}

    RULES:
    - Be concise
    - Use clear language
    - Do not invent information
    """

def build_rag_prompt(context: str, question: str) -> str: 
    return f""" 
You are a helpful AI assistant. 
Use ONLY the context below to answer the question. 
CONTEXT: 
{context} 
QUESTION: 
{question} 
RULES: 
- Do not invent information - If answer is missing, say "I don't know"
- If answer is missing, say "I don't know"
"""