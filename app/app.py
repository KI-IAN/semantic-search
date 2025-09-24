import gradio as gr
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz # PyMuPDF
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
import os
from config import GOOGLE_API_KEY, NVIDIA_API_KEY, CHROMA_DIR



def setup_nvidia_embedding_model():

    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    nvidia_embedding_model = "nvidia/nv-embed-v1"
    embedding_model = NVIDIAEmbeddings(model=nvidia_embedding_model)
    
    return embedding_model

def setup_google_gemini_embedding_model():
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    gemini_embedding_model = "models/gemini-embedding-001"
    embedding_model = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)

    return embedding_model

vectorstore = None

def answer_with_llm(query, retrieved_docs, model_name, model_provider):
    if not retrieved_docs:
        return "No relevant information found to answer your question."

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    prompt = f""" 
    You are an expert assistant. Use the following context to answer the user's question. 
    If you do not find or know the answer, do not hallucinate, do not try to generate fake answers.
    If no Context is given, simply state "No relevant information found to answer your question."
    
    Context: 
    {context}

    Question:
    {query}
    
    Answer:
    
    """
    
    llm = init_chat_model(model_name, model_provider=model_provider)
    response = llm.invoke(prompt)
    return response.content


def read_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text


def process_pdf(file):
    global vectorstore

    if not file:
        return "Error: No file uploaded."
    
    text = read_pdf(file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    #Create Chroma DB
    embedding_model = setup_nvidia_embedding_model()
    # embedding_model = setup_google_gemini_embedding_model()
    
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=CHROMA_DIR )
    # vectorstore.persist()
    
    return f"PDF processed and stored with {len(docs)} chunks."


def search_query(query):
    if not vectorstore:
        error_message = "Error: No vectorstore found. Please upload and process a PDF first."
        return error_message, error_message

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents(query)
    # results = retriever.similarity_search(query, k=2)

    # return "\n\n-------------\n\n".join(doc.page_content for doc in results)

    semantic_search_response = "\n\n-------------\n\n".join(doc.page_content for doc in results)

    # model_name = "gemini-2.5-pro"
    # model_provider = "google_genai"

    model_name = "bytedance/seed-oss-36b-instruct"
    model_provider = "nvidia"
    
    llm_answer = answer_with_llm(query, results, model_name=model_name, model_provider=model_provider)
    return semantic_search_response, llm_answer

#Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Semantic Search App (Langchain + ChromaDb + NVidia LLM API)")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF (max 5mb)", type="binary", file_types=[".pdf"])
        process_btn = gr.Button("Process PDF")
        
    status = gr.Textbox(label="Status")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query")
        search_btn = gr.Button("Semantic Search")
        
    with gr.Row():
        semantic_search_response = gr.Textbox(label="Semantic Search Response", lines=10, show_copy_button=True)
        llm_response = gr.Textbox(label="LLM Response", lines=10, show_copy_button=True)

    process_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status)
    search_btn.click(fn=search_query, inputs=query_input, outputs=(semantic_search_response, llm_response))
    
    
demo.launch(server_name="0.0.0.0")



    
    