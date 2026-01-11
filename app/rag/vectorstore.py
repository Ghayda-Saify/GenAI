import chromadb 
from chromadb.config import Settings 
from app.rag.embeddings import embed_text 
client = chromadb.Client( 
    Settings(persist_directory="./chroma_db") 
) 
collection = client.get_or_create_collection( 
    name="documents", 
    embedding_function=lambda texts: [embed_text(t) for t in texts] 
) 
def add_documents(texts: list[str]): 
    collection.add( 
        documents=texts, 
        ids=[str(i) for i in range(len(texts))] 
    ) 
def query_documents(query: str, n_results=3): 
    return collection.query( 
        query_texts=[query], 
        n_results=n_results 
    ) 
