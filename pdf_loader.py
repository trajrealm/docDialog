from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    return split_docs
