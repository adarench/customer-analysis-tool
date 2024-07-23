# src/embedding.py

import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_chunks(chunks):
    if not chunks:
        raise ValueError("Chunks list is empty.")
    embeddings = model.encode(chunks)
    return embeddings

def find_similar_chunks(query_embedding, chunk_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    indices = np.argsort(similarities)[::-1]
    scores = similarities[indices]
    return indices, scores
