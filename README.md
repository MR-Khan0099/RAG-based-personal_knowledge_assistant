# Personal Knowledge Assistant

A powerful, user-friendly chatbot that answers questions about your personal documents (PDFs, DOCX, Markdown, TXT, and more) using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). Organize your digital clutter and get instant, context-aware answers from your files!

---

## 🚀 Features
- **Multi-file Support:** Ingests PDFs, Word documents, Markdown, and plain text files.
- **Advanced RAG Workflow:** Retrieves relevant information from your files and uses an LLM to generate coherent answers.
- **Vector Database Flexibility:** Choose between Chroma (local) or Pinecone (cloud) for vector storage.
- **Modern Web UI:** Streamlit-based chat interface for easy interaction.
- **Chat History Persistence:** Saves and loads your chat history automatically.
- **LLM Flexibility:** Easily switch between OpenAI models (GPT-3.5, GPT-4) and extend to others (Gemini, Claude, HuggingFace, etc.).

---

## 🏗️ Architecture
- **Document Ingestion:** Loads and splits documents, creates embeddings, and stores them in a vector database.
- **Retrieval:** Finds relevant chunks using vector search.
- **LLM Integration:** Uses the retrieved context to answer user questions via an LLM.
- **Web Interface:** Streamlit app for chatting, model selection, and chat history.

---

## 📂 Supported File Types
- PDF (`.pdf`)
- Word (`.docx`)
- Markdown (`.md`)
- Plain Text (`.txt`)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd personal_knowledge_assistant
```

### 2. Create and Activate a Virtual Environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with the following:
```env
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=personal-knowledge-assistant
VECTOR_DB=pinecone  # or chroma
```
- For Pinecone, set your region in `ingest.py` and `app.py` (e.g., `us-west-2`).
- If you only want local storage, set `VECTOR_DB=chroma` and ignore Pinecone variables.

### 5. Add Your Documents
Place your files in the `data/` directory. Supported types: PDF, DOCX, MD, TXT.

### 6. Ingest Your Documents
```bash
python ingest.py
```
This will process your files and build the vector database.

### 7. Run the Streamlit App
```bash
streamlit run app.py
```
Open the provided local URL in your browser to start chatting!

---

## 🧑‍💻 Usage
- **Ask questions** about your documents in natural language.
- **Switch models** and vector DBs in the sidebar.
- **Clear chat history** with one click.
- **Add new documents** by placing them in `data/` and re-running `ingest.py`.

---

## 🛠️ Customization & Extensibility
- **LLMs:** Easily extend to Gemini, Claude, or HuggingFace models by modifying `app.py`.
- **Vector DBs:** Add support for Qdrant, FAISS, or others via LangChain.
- **Retrieval:** Try hybrid search or advanced retrievers for better results.
- **UI:** Enhance the Streamlit interface for uploads, document management, or user authentication.

---

## 📑 Example .env File
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west-2
PINECONE_INDEX_NAME=personal-knowledge-assistant
VECTOR_DB=pinecone
```

---

## 🤝 Credits
- Built with [LangChain](https://github.com/langchain-ai/langchain), [Streamlit](https://streamlit.io/), and [Pinecone](https://www.pinecone.io/).

---

## 📬 Feedback & Contributions
Pull requests and suggestions are welcome! If you find this useful or have ideas for improvement, please open an issue or PR. 