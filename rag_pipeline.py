from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from ingest import load_and_chunk_pdf
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/sample_pdfs/example.pdf"

def create_vector_store():
    # 1. Load and chunk PDF
    chunks = load_and_chunk_pdf(PDF_PATH)

    # 2. Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # 3. Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("Vector store created successfully")

    return vector_store


def similarity_search(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    vs = create_vector_store()

    query = "What is artificial intelligence?"
    docs = similarity_search(query, vs)

    print("\n--- Retrieved Chunks ---\n")
    for i, doc in enumerate(docs):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print("-" * 40)