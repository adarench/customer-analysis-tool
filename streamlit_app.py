# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG System", page_icon="📈")

st.title("Upload and Chat with Your Data")

directory_path = st.text_input("Enter the directory path for documents (file or directory):")

if st.button("Process Documents"):
    response = requests.post('http://localhost:5001/process_documents', json={'directory_path': directory_path})
    if response.status_code == 200:
        result = response.json()
        chunks = result['chunks']
        embeddings = result['embeddings']
        st.write("Documents processed. Embeddings created.")
    else:
        st.write(f"Error processing documents: {response.text}")


chat_input = st.text_input("Enter your query or chat message:")

if st.button("Chat"):
    with st.spinner("Chatting with your data..."):
        response = requests.post('http://localhost:5001/chat', json={'user_input': chat_input}, stream=True)
        if response.status_code == 200:
            st.write("Chat Response:")
            chat_response = ""
            for line in response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8')
                    if line_decoded == "[DONE]":
                        break
                    chat_response += line_decoded
                    st.write(chat_response)
        else:
            st.write(f"Error chatting: {response.text}")
