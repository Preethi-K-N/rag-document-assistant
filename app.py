import streamlit as st
from ingest import load_and_chunk_pdf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
import tempfile

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📄 RAG-powered Document Q&A Assistant")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

@st.cache_resource
def create_vector_store_from_pdf(pdf_path):
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def ask_question(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = """
You are an AI assistant answering strictly from the given context.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(prompt.format(context=context, question=query))
    return response.content, docs


if uploaded_file:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.success("PDF uploaded and processed successfully")
        vector_store = create_vector_store_from_pdf(tmp_path)
        user_question = st.text_input("Ask a question about the document")

        if user_question:
            answer, sources = ask_question(user_question, vector_store)
            st.subheader("✅ Answer")
            st.write(answer)

            with st.expander("📌 Source Chunks"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}**")
                    st.write(doc.page_content[:400])
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)