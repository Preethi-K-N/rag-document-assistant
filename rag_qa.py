from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from ingest import load_and_chunk_pdf
from dotenv import load_dotenv
import os

load_dotenv()

PDF_PATH = "data/sample_pdfs/example.pdf"

def build_vector_store():
    chunks = load_and_chunk_pdf(PDF_PATH)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_rag_answer(query, vector_store, k=3):
    # 1. Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # 2. Create prompt
    prompt_template = """
You are an AI assistant answering questions strictly using the provided context.
If the answer is not present in the context, say "I don't know based on the given document."

Context:
{context}

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 3. Initialize LLM
    llm = ChatOpenAI(temperature=0)

    # 4. Generate answer
    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    return response.content, retrieved_docs


if __name__ == "__main__":
    vector_store = build_vector_store()

    user_question = "What is artificial intelligence?"
    answer, sources = get_rag_answer(user_question, vector_store)

    print("\nAnswer:\n", answer)
    print("\nSources:\n")
    for i, doc in enumerate(sources):
        print(f"Source {i+1}:")
        print(doc.page_content[:200])
        print("-" * 50)