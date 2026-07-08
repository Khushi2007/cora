import os
import gradio as gr

from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# Embedding Model
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# Text Splitter
# ==================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


# ==================================================
# ChromaDB
# ==================================================

CHROMA_DIR = "./chroma_store"

os.makedirs(CHROMA_DIR, exist_ok=True)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


# ==================================================
# Current Uploaded PDFs
# ==================================================

uploaded_documents = []


# ==================================================
# Index Documents
# ==================================================

def index_documents(
    pdf_paths: list[str],
    progress=gr.Progress()
) -> str:
    """
    Clears the existing Chroma collection and indexes
    all uploaded PDF documents.
    """

    global vectorstore
    global uploaded_documents

    if not pdf_paths:
        return "No PDFs selected."

    try:

        progress(0.05, desc="Preparing database...")

        # Clear previous indexed documents
        clear_documents()

        uploaded_documents = []

        documents = []

        progress(0.20, desc="Loading PDFs...")

        for pdf in pdf_paths:

            loader = PyPDFLoader(pdf)
            pages = loader.load()

            filename = os.path.basename(pdf)

            uploaded_documents.append(filename)

            for page in pages:
                page.metadata["source"] = filename

            documents.extend(pages)

        progress(0.45, desc="Splitting into chunks...")

        chunks = splitter.split_documents(documents)

        progress(0.70, desc="Creating embeddings...")

        vectorstore.add_documents(chunks)

        progress(1.0, desc="Done!")

        return (
            f"✅ Successfully indexed "
            f"{len(uploaded_documents)} PDF(s)\n"
            f"({len(chunks)} searchable chunks)"
        )

    except Exception as e:
        return f"❌ Indexing failed:\n{e}"


# ==================================================
# Search Tool
# ==================================================

@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded PDF documents.

    Use this tool whenever the user asks about:

    • uploaded PDFs
    • uploaded documents
    • reports
    • lecture notes
    • books
    • manuals
    • stories
    • articles
    • "according to the uploaded PDF..."
    • "summarize the document"
    • "what does page X say?"

    If no PDFs have been uploaded,
    politely explain that document search
    is unavailable.
    """

    global vectorstore

    if len(uploaded_documents) == 0:
        return (
            "No PDF documents have been uploaded yet. "
            "Please ask the user to upload one first."
        )

    try:

        docs = vectorstore.similarity_search(
            query,
            k=4,
        )

        if not docs:
            return "No relevant information was found."

        results = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            results.append(
                f"""
Source: {source}
Page: {page + 1}

{doc.page_content}
"""
            )

        return "\n" + "\n--------------------------\n".join(results)

    except Exception as e:
        return f"Document search failed: {e}"


# ==================================================
# Clear Documents
# ==================================================

def clear_documents():

    global uploaded_documents

    try:
        ids = vectorstore.get()["ids"]

        if ids:
            vectorstore.delete(ids=ids)

    except Exception:
        pass

    uploaded_documents.clear()

    return "Documents cleared successfully."


# ==================================================
# Uploaded Document List
# ==================================================

def list_documents():
    """
    Returns a formatted list of
    currently uploaded PDFs.
    """

    if not uploaded_documents:
        return "No PDFs uploaded."

    lines = ["### Uploaded PDFs\n"]

    for pdf in uploaded_documents:
        lines.append(f"✅ {pdf}")

    return "\n".join(lines)