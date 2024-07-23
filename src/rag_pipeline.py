# rag_pipeline.py
from src.chunking import AgenticChunker
from src.embedding import embed_chunks
from src.database import store_embeddings as db_store_embeddings, retrieve_similar_embeddings as db_retrieve_similar_embeddings
from src.utils import read_files_in_directory
from langchain_openai import ChatOpenAI

def process_documents(directory_path, llm):
    chunker = AgenticChunker(openai_api_key=llm.openai_api_key)
    combined_text = read_files_in_directory(directory_path)
    chunker.add_propositions(combined_text)
    
    # Collect all chunks
    chunks = [prop['proposition'] for chunk in chunker.chunks.values() for prop in chunk['propositions']]
    
    # Create embeddings
    embeddings = embed_chunks(chunks)
    
    return chunks, embeddings

def get_query_embedding(query, llm):
    query_embedding = embed_chunks([query])
    return query_embedding[0]

def store_embeddings(chunks, embeddings):
    metadata = [{'text': chunk} for chunk in chunks]
    db_store_embeddings(embeddings, metadata)

def retrieve_similar_embeddings(query_embedding, top_k=5):
    # Ensure the query_embedding is converted to a list
    query_embedding = query_embedding.tolist()
    return db_retrieve_similar_embeddings(query_embedding, top_k)
