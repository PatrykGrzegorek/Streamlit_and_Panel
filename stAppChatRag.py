import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
from uuid import uuid4
import torch
import time
from datetime import datetime

# Ustawienia strony Streamlit
st.set_page_config(page_title="AI Chatbot", layout="centered")

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja modelu embeddingów
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inicjalizacja Chroma vector store
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
)

# Sprawdzenie, czy vector_store jest pusty i dodanie przykładowych dokumentów
if not vector_store._collection.count():
    # Tworzenie przykładowych dokumentów
    documents = [
        Document(page_content="The capital of France is Paris.", metadata={"source": "app", "timestamp": time.time()}, id=str(uuid4())),
        Document(page_content="The Eiffel Tower is located in Paris.", metadata={"source": "app", "timestamp": time.time()}, id=str(uuid4())),
        Document(page_content="The Louvre is the world's largest art museum and a historic monument in Paris.", metadata={"source": "app", "timestamp": time.time()}, id=str(uuid4())),
        Document(page_content="The French Revolution began in 1789.", metadata={"source": "app", "timestamp": time.time()}, id=str(uuid4())),
        Document(page_content="Napoleon Bonaparte was a French military leader and emperor.", metadata={"source": "app", "timestamp": time.time()}, id=str(uuid4())),
    ]
    # Dodawanie dokumentów do vector_store
    vector_store.add_documents(documents)

# Ustawienie retrievera
retriever = vector_store.as_retriever()

# Inicjalizacja modelu Hugging Face i tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Ustawienie pipeline'u
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, device=device.index if device.type == "cuda" else -1)

# Inicjalizacja LLM z pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Ustawienie łańcucha RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Inicjalizacja historii czatu
if "history" not in st.session_state:
    st.session_state.history = []

# Funkcja do dodawania wiadomości do historii
def add_message(sender, message):
    st.session_state.history.append({"sender": sender, "message": message})

# Interfejs Streamlit
st.title("AI Chatbot")
st.write("Ask any question and AI will answer.")

col1, col2 = st.columns([2, 1])
with col1:
    # Pole tekstowe dla użytkownika
    user_input = st.text_input("Your message:")

    # Przyciski akcji
    col11, col12, col13 = st.columns(3)
    with col11:
        send_button = st.button("Send")
    with col12:
        clear_chat_button = st.button("Clear chat")
    with col13:
        clear_docs_button = st.button("Clear documents")

    # Obsługa przycisku "Send"
    if send_button:
        if user_input:
            add_message("user", user_input)

            # Użycie łańcucha QA do uzyskania odpowiedzi
            response = qa_chain({"query": user_input})

            answer = response["result"]
            source_documents = response["source_documents"]

            add_message("assistant", answer)

    # Obsługa przycisku "Clear chat"
    if clear_chat_button:
        st.session_state.history = []

    # Obsługa przycisku "Clear documents"
    if clear_docs_button:
        doc_ids = vector_store._collection.get()['ids']

        if doc_ids:
            vector_store._collection.delete(ids=doc_ids)
            st.success("All documents have been deleted from the vector store.")
        else:
            st.warning("No documents to delete.")


    # Formularz do dodawania nowego dokumentu
    st.write("### Add a new document to the vector store:")
    with st.form(key='add_document_form'):
        new_doc_content = st.text_area('Document content:')
        new_doc_metadata = st.text_input('Metadata (optional):')
        submit_doc = st.form_submit_button('Add Document')

    if submit_doc and new_doc_content:
        new_document = Document(
            page_content=new_doc_content,
            metadata={"source": new_doc_metadata if new_doc_metadata else "unknown", "timestamp": time.time()},
            id=str(uuid4())
        )
        vector_store.add_documents([new_document])
        st.success('Document added successfully!')


    # Wyświetlanie czatu
    st.write("### Chat:")
    chat_container = st.container()
    with chat_container:
        chat_history_css = """
            <style>
            .chat-history {
                max-height: 400px;
                overflow-y: auto;
                padding-right: 20px;
                scrollbar-width: thin;
                scrollbar-color: #555 #33372C;
            }
    
            .chat-history::-webkit-scrollbar {
                width: 8px;
            }
    
            .chat-history::-webkit-scrollbar-track {
                background: #33372C;
            }
    
            .chat-history::-webkit-scrollbar-thumb {
                background-color: #555;
                border-radius: 10px;
                border: 2px solid #33372C;
            }
    
            .chat-history::-webkit-scrollbar-thumb:hover {
                background-color: #777;
            }
    
            .chat-history::-webkit-scrollbar-button {
                display: none;
            }
            </style>
        """
        chat_content = "<div class='chat-history'>"
        for chat in st.session_state.history:
            if chat["sender"] == "user":
                chat_content += f"""
                <div style='font-family: "Roboto", sans-serif; font-style: normal; text-align: right; background-color: #654520; color: white; padding: 10px; border-radius: 10px; margin: 5px;'>
                    <strong>You:</strong> {chat['message']}
                </div>
                """
            else:
                chat_content += f"""
                <div style='font-family: "Roboto", sans-serif; font-style: normal; text-align: left; background-color: #33372C; color: white; padding: 10px; border-radius: 10px; margin: 5px;'>
                    <strong>Assistant:</strong> {chat['message']}
                </div>
                """
        chat_content += "</div>"

        scroll_js = """
            <script>
            var chatDiv = document.getElementsByClassName('chat-history')[0];
            chatDiv.scrollTop = chatDiv.scrollHeight;
            </script>
        """

        st.components.v1.html(chat_history_css + chat_content + scroll_js, height=400)

# Wyświetlanie dokumentów w vector_store
with col2:
    st.write("### Documents in Vector Store:")

    # Pobieranie wszystkich dokumentów z vector_store
    all_documents = vector_store._collection.get()['documents']
    all_metadatas = vector_store._collection.get()['metadatas']

    # Łączenie dokumentów z metadanymi i sortowanie na podstawie timestamp
    documents_with_metadata = sorted(
        zip(all_documents, all_metadatas),
        key=lambda x: x[1].get('timestamp', 0),
        reverse=True  # Najnowsze dokumenty na górze
    )

    # Wyświetlanie od najnowszych do najstarszych
    for idx, (doc_content, metadata) in enumerate(documents_with_metadata):
        # Pobranie timestamp z metadanych
        timestamp = metadata.get('timestamp', None)

        # Konwersja timestamp na czytelną datę i godzinę
        if timestamp:
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted_time = "Unknown"

        st.write(f"**Document:** {doc_content}")
        st.write(f"**Source:** {metadata.get('source', None)}")
        st.write(f"**Date:** {formatted_time}")
        st.write("---")
