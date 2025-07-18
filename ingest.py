import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, Pinecone as CommunityPinecone
from langchain_openai import OpenAIEmbeddings
import pinecone

# Pinecone official import
import pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
data_path = "data"
db_path = "vector_db"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
VECTOR_DB = os.getenv("VECTOR_DB", "chroma").lower()  # 'chroma' or 'pinecone'
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "personal-knowledge-assistant")


def load_all_documents():
    """Load all supported documents from the data directory."""
    documents = []
    # PDF
    pdf_loader = PyPDFDirectoryLoader(data_path)
    documents.extend(pdf_loader.load())
    # DOCX
    for file in os.listdir(data_path):
        if file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
    # Markdown
    for file in os.listdir(data_path):
        if file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
    # TXT
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
    return documents

def ingest_documents():
    # 1. Load docs from the data directory
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents.")

    # 2. Split docs into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create a vector store from the chunks
    print("Creating embeddings and storage in vector DB....")
    embeddings = OpenAIEmbeddings()

    if VECTOR_DB == "pinecone":
        if not PINECONE_API_KEY or not PINECONE_ENV:
            raise ValueError("Pinecone API key and environment must be set in .env")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Example: check/create index
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')  # adjust as needed
            )
        # Use pc.Index(PINECONE_INDEX_NAME) as the index object
        db = CommunityPinecone.from_documents(
            chunks,
            embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        print(f"Successfully created Pinecone vector DB: {PINECONE_INDEX_NAME}")
    else:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=db_path
        )
        db.persist()
        print(f"Successfully created Chroma vector DB at {db_path}")

if __name__ == "__main__":
    ingest_documents()











