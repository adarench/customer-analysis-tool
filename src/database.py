# database.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer 
import uuid

client = chromadb.PersistentClient(
    path="./.chromadb",
    settings=Settings(anonymized_telemetry=False)
)

def store_embeddings(embeddings, metadata):
    if "rag_embeddings" in [col.name for col in client.list_collections()]:
        collection = client.get_collection(name="rag_embeddings")
    else:
        collection = client.create_collection(name="rag_embeddings")

    ids = [str(uuid.uuid4()) for _ in embeddings]
    documents = [meta['text'] for meta in metadata]
    # Convert embeddings to list of lists
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    collection.add(ids=ids, documents=documents, embeddings=embeddings_list)

def retrieve_similar_embeddings(query_embedding, top_k=5):
    if "rag_embeddings" not in [col.name for col in client.list_collections()]:
        raise ValueError("The collection 'rag_embeddings' does not exist.")
    
    collection = client.get_collection(name="rag_embeddings")
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    return [{"text": res, "score": score} for res, score in zip(results['documents'][0], results['distances'][0])]

