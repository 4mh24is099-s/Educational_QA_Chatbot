import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file):
    pdf_file.seek(0)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""

    for page in doc:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text + "\n"

    return text


def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_text(text)


def create_vector_store(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    # Cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    return index


def search(question, vector_store, chunks, k=6):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = vector_store.search(
        query_embedding.astype("float32"),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        if idx != -1:
            retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)