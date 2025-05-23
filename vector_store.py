from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

def create_vector_store(docs, persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore
