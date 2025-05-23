import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler    

load_dotenv()

def build_qa_chain(vectorstore, callbacks: list[BaseCallbackHandler] = None):
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.3,
        streaming=True,
        callbacks=callbacks or []
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain