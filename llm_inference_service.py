
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# Initialize Langfuse client
# This block sets up the Langfuse callback handler for LangChain.
# It initializes the Langfuse client and creates a CallbackHandler instance
# only if the required API keys are available. The handler is then added to
# a list of callbacks that can be passed to LLM invocations for tracing.
langfuse_callback_handler = None
callbacks = []

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse_callback_handler = CallbackHandler()
    callbacks.append(langfuse_callback_handler)

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
    If no Context is given or you can't find or generate any relevant information to answer the question, simply state "No relevant information found to answer your question."
    
    Context: 
    {context}

    Question:
    {query}
    
    Answer:
    
    """
    
    llm = init_chat_model(model_name, model_provider=model_provider)
    response = llm.invoke(prompt, config={"callbacks": callbacks})
    return response.content