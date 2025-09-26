import os
"""
Configuration module for loading environment variables and secrets.
- Loads environment variables from a `.env` file located in the project root if it exists else loads from Hugging Face's Secrets Tab.
- Provides access to the following secrets:
    - GOOGLE_API_KEY: API key for Gemini LLM.
    - NVIDIA_API_KEY: API key for NVIDIA LLM.
    - CHROMA_DIR: Directory path for Chroma DB (defaults to './chroma_db').
- Prints warnings if required API keys are not set.
"""
from dotenv import load_dotenv

#Load .env only if running locally
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    
    
# Access Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")        
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db") 

if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY is not set. Gemini LLM API may fail.")
    
if not NVIDIA_API_KEY:
    print("⚠️ Warning: NVIDIA_API_KEY is not set. NVIDIA LLM API may fail.")
