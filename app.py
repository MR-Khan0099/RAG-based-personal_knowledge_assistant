import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma, Pinecone as CommunityPinecone
from langchain.chains import RetrievalQA
import pinecone
import json

load_dotenv()

# --- Config ---
DB_PATH = "vector_db"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "personal-knowledge-assistant")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma").lower()  # 'chroma' or 'pinecone'
CHAT_HISTORY_FILE = "chat_history.json"

# --- Utility: Chat History Persistence ---
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# --- Modular LLM/Embedding Selection ---
def get_llm(model_name="gpt-3.5-turbo", temperature=0):
    return ChatOpenAI(model=model_name, temperature=temperature)

def get_embeddings():
    return OpenAIEmbeddings()

# --- Modular Retriever Selection ---
def get_retriever(vector_db_backend, embeddings):
    if vector_db_backend == "pinecone":
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        db = CommunityPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
    else:
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# --- Streamlit GUI ---
st.set_page_config(page_title="Personal Knowledge Assistant", layout="wide")
st.title("📚 Personal Knowledge Assistant")

# Sidebar: Model and DB selection
with st.sidebar:
    st.header("Settings")
    vector_db_backend = st.selectbox("Vector DB Backend", ["chroma", "pinecone"], index=0 if VECTOR_DB=="chroma" else 1)
    model_name = st.selectbox("LLM Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)
    if st.button("Clear Chat History"):
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.session_state["chat_history"] = []
        st.rerun()

# Load chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history()

# File upload (for future extension, not used in this script)
st.sidebar.markdown("---")
st.sidebar.file_uploader("Upload new document (PDF, DOCX, MD, TXT)", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True, disabled=True)
st.sidebar.info("To add new documents, place them in the 'data' folder and re-run the ingestion script.")

# Set up LLM, embeddings, retriever, and QA chain
llm = get_llm(model_name, temperature)
embeddings = get_embeddings()
retriever = get_retriever(vector_db_backend, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat interface
st.markdown("---")
st.subheader("Ask a question about your documents:")
user_input = st.text_input("Your question:", key="user_input")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_input)
        st.session_state["chat_history"].append({"question": user_input, "answer": answer})
        save_chat_history(st.session_state["chat_history"])
        st.rerun()

# Display chat history
if st.session_state["chat_history"]:
    st.markdown("### Chat History")
    for i, entry in enumerate(reversed(st.session_state["chat_history"])):
        st.markdown(f"**Q{i+1}:** {entry['question']}")
        st.markdown(f"> {entry['answer']}")
        st.markdown("---")
else:
    st.info("No chat history yet. Start by asking a question!")






























