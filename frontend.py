import gradio as gr
from rag_pipeline import build_qa_chain
from pdf_loader import load_and_split_pdfs
from vector_store import create_vector_store
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue, Empty
import threading
import time


# Load and process PDFs
docs = load_and_split_pdfs("data")
vectorstore = create_vector_store(docs)
# qa_chain = build_qa_chain(vectorstore)

class GradioStreamingCallback(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

def chat_with_pdf(message, history):
    q = Queue()
    callback = GradioStreamingCallback(q)
    
    qa_chain = build_qa_chain(vectorstore, callbacks=[callback])

    def generate():
        thread = threading.Thread(target=qa_chain, args=({"query": message},))
        thread.start()
        while thread.is_alive() or not q.empty():
            try:
                token = q.get(timeout=0.1)
                yield token
            except Empty:
                continue
        thread.join()        

    return generate 
    # sources = result.get("source_documents", [])
    
    # if sources:
    #     source_texts = "\n\n".join(doc.page_content[:300] for doc in sources[:2])
    #     answer += f"\n\n📚 Sources:\n{source_texts}"

    return answer
def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 📄 AskPDFly\nChat with your PDF knowledge base.")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask something about your PDFs...", label="Your Question")
        clear = gr.Button("Clear Chat")

        def respond(message, history):
            response = ""
            for token in chat_with_pdf(message, history)():
                response += token
                yield history + [(message, response)]

        msg.submit(respond, [msg, chatbot], chatbot, show_progress=True)
        clear.click(lambda: [], None, chatbot)

    demo.launch()