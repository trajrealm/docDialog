from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def load_vector_store(persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
