from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/sample_pdfs/example.pdf"

def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")

    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_pdf(PDF_PATH)
    print("\n--- Sample Chunk ---\n")
    print(chunks[0].page_content)