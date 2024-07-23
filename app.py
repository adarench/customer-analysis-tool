# app.py
from flask import Flask, request, jsonify, Response
from src.rag_pipeline import process_documents, store_embeddings, retrieve_similar_embeddings, get_query_embedding
from langchain_openai import ChatOpenAI
import os
import numpy as np
from openai import OpenAI

client = OpenAI()

app = Flask(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=api_key)

@app.route('/process_documents', methods=['POST'])
def process_docs():
    try:
        data = request.json
        directory_path = data['directory_path']
        chunks, embeddings = process_documents(directory_path, llm)
        if not chunks:
            return jsonify({'error': 'No chunks were created from the documents.'}), 500
        store_embeddings(chunks, embeddings)
        return jsonify({'chunks': chunks, 'embeddings': embeddings.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.json
        query = data['query']
        query_embedding = get_query_embedding(query, llm)
        return jsonify({'embedding': query_embedding.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.json
        query_embedding = np.array(data['query_embedding'])
        top_k = data.get('top_k', 5)
        results = retrieve_similar_embeddings(query_embedding, top_k)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data['user_input']

        # Get query embedding
        query_embedding = get_query_embedding(user_input, llm)

        # Retrieve similar documents
        results = retrieve_similar_embeddings(np.array(query_embedding))

        # Compile document context
        document_context = "\n\n".join([result['text'] for result in results])

        # Generate response from LLM including the document context
        llm_prompt = f"User query: {user_input}\n\nRelevant documents:\n{document_context}\n\nResponse:"

        # Make synchronous request to OpenAI API with streaming response
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": llm_prompt}
        ],
        stream=True)

        def generate():
            for chunk in response:
                if 'choices' in chunk and chunk.choices[0].get('delta', {}).content:
                    yield f"{chunk.choices[0].delta.content}"
            yield "[DONE]"

        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
