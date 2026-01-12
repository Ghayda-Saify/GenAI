from app.rag.vectorstore import query_documents

def retrieve_context(query: str) -> str:
    results = query_documents(query)
    docs = results["documents"][0]  # list of docs
    return "\n".join(docs)
