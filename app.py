
"""
This module implements a semantic search application using LangChain, ChromaDB, and NVIDIA/Google LLM APIs.
It provides functionality to process PDF documents, split and embed their contents, store embeddings in a vector database,
and perform semantic search with LLM-powered answers via a Gradio UI.
Main Components:
---------------
- setup_nvidia_embedding_model: Initializes and returns an NVIDIA embedding model for semantic search.
- setup_google_gemini_embedding_model: Initializes and returns a Google Gemini embedding model for text embeddings.
- answer_with_llm: Uses a language model to generate answers to user queries based on retrieved document context.
- read_pdf: Extracts text from a PDF file.
- process_pdf: Reads, splits, embeds, and stores PDF content in a Chroma vector database.
- search_query: Performs semantic search and generates LLM answers based on user queries.
- Gradio UI: Provides an interactive interface for uploading PDFs, processing documents, and querying the semantic search system.
Global Variables:
-----------------
- vectorstore: Stores the Chroma vector database instance for semantic search.
Configuration:
--------------
- Requires API keys for Google and NVIDIA, and a directory path for ChromaDB persistence, imported from config.py.
Usage:
------
1. Upload and process a PDF to initialize the vectorstore.
2. Enter a query to perform semantic search and receive contextually relevant answers from an LLM.
Dependencies:
-------------
- gradio
- langchain_nvidia_ai_endpoints
- langchain_text_splitters
- fitz (PyMuPDF)
- langchain.docstore.document
- langchain.vectorstores
- langchain_google_genai
- langchain.chat_models
- os
- config (GOOGLE_API_KEY, NVIDIA_API_KEY, CHROMA_DIR)
"""

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
    """
    Sets up and returns an NVIDIA embedding model for semantic search.
    This function configures the NVIDIA API key in the environment,
    initializes the NVIDIA embedding model using the specified model name,
    and returns the embedding model instance.
    Returns:
        NVIDIAEmbeddings: An instance of the NVIDIA embedding model.
    Raises:
        NameError: If NVIDIA_API_KEY is not defined.
        ImportError: If NVIDIAEmbeddings is not available.
    """

    os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
    nvidia_embedding_model = "nvidia/nv-embed-v1"
    embedding_model = NVIDIAEmbeddings(model=nvidia_embedding_model)
    
    return embedding_model

def setup_google_gemini_embedding_model():
    """
    Sets up and returns a Google Gemini embedding model for generating text embeddings.
    This function configures the environment with the required Google API key and initializes
    the Gemini embedding model using the specified model name.
    Returns:
        GoogleGenerativeAIEmbeddings: An instance of the Gemini embedding model for text embeddings.
    """
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    gemini_embedding_model = "models/gemini-embedding-001"
    embedding_model = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)

    return embedding_model

vectorstore = None

def answer_with_llm(query, retrieved_docs, model_name, model_provider):
    """Generates an answer to a user's question using a language model and retrieved documents as context.
    Args:
        query (str): The user's question to be answered.
        retrieved_docs (list): A list of document objects containing relevant context. Each document should have a 'page_content' attribute.
        model_name (str): The name of the language model to use for generating the answer.
        model_provider (str): The provider of the language model.
    Returns:
        str: The generated answer from the language model. If no relevant documents are found, returns a default message indicating no relevant information.
    """
    
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
    """
    Extracts and returns the text content from a PDF file.

    Args:
        file (bytes): The PDF file as a byte stream.

    Returns:
        str: The extracted text from all pages of the PDF, separated by newlines.
    """
    doc = fitz.open(stream=file, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text


def process_pdf(file):
    """
    Processes a PDF file by reading its contents, splitting the text into chunks, embedding the chunks, 
    and storing them in a Chroma vector database.
    Args:
        file: The uploaded PDF file to process.
    Returns:
        str: A message indicating the result of the processing, or an error message if no file is uploaded.
    Side Effects:
        Updates the global `vectorstore` variable with the new Chroma vector database.
    """
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
    """
    Performs a semantic search on the provided query using a vectorstore and returns relevant document contents
    along with an answer generated by a large language model (LLM).
    Args:
        query (str): The search query string.
    Returns:
        tuple:
            - semantic_search_response (str): Concatenated contents of the top relevant documents separated by delimiters.
            - llm_answer (str): The answer generated by the LLM based on the query and retrieved documents.
    Raises:
        Returns an error message tuple if the vectorstore is not initialized.
    """
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
    gr.Markdown("# 🧠 Semantic Search App (Langchain + ChromaDb + Nvidia NIM API)")
    
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



    
    