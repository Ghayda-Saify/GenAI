import chromadb
from chromadb.config import Settings
from app.rag.embeddings import embed_text

# ---- Chroma expects an EmbeddingFunction object with __call__(self, input) ----
class Embedder:
    def __call__(self, input):
        # input is typically a list[str]
        return [embed_text(t) for t in input]

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=Embedder()
)

def add_documents(texts: list[str]):
    # Make IDs unique across multiple calls (avoid overwriting / duplicates)
    start = collection.count()
    ids = [str(start + i) for i in range(len(texts))]

    collection.add(
        documents=texts,
        ids=ids
    )

def query_documents(query: str, n_results=3):
    return collection.query(
        query_texts=[query],
        n_results=n_results
    )
