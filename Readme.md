---
title: Semantic Search App
emoji: 📄🔗🧠❓🔗🤖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.46.1
app_file: app/app.py
pinned: false
---

# Semantic Search App (📄 → 🔗 → 🧠 → ❓ → 🔗 → 🤖)

Upload a PDF, ask questions, and get context-aware answers powered by LangChain, ChromaDB, and NVIDIA/Google LLMs — all wrapped in a clean Gradio interface.

🔗 **Live Demo**: [https://huggingface.co/spaces/frkhan/semantic-search-app](#)

---

### 🚀 Features

- 📄 Upload and process PDF documents  
- 🔍 Perform semantic search using vector embeddings  
- 🤖 Get answers from powerful LLMs (NVIDIA or Google Gemini)  
- 🧠 Uses LangChain + ChromaDB for retrieval  
- 🧰 Docker-ready and Hugging Face Spaces–compatible  

---

### 🛠️ Tech Stack

| Component        | Purpose                                 |
|------------------|-----------------------------------------|
| LangChain        | Orchestration of embedding + LLM calls  |
| ChromaDB         | Vector database for semantic retrieval  |
| NVIDIA / Gemini  | Embedding + LLM APIs                    |
| Gradio           | Interactive UI                          |
| Docker           | Containerized deployment                |

---

## 📦 Installation

### Option 1: Run Locally

```bash
git clone https://github.com/KI-IAN/semantic-search.git
cd semantic-search
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Create a .env file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key
NVIDIA_API_KEY=your_nvidia_api_key
CHROMA_DIR=./chroma_db
```

Then run:

```bash
python app/app.py
```

---

### Option 2: Run with Docker

```bash
docker-compose up --build
```

Access the app at http://localhost:1200

---

### Option 3: Deploy on Hugging Face Spaces

Create a new Space → choose Gradio as the SDK

Upload your project files (including app/, Dockerfile, requirements.txt, .env)

Set Secrets in the “Secrets” tab:

GOOGLE_API_KEY

NVIDIA_API_KEY

(Optional) CHROMA_DIR (defaults to ./chroma_db)

Hugging Face will auto-detect and launch the app via Gradio

---

## 🔑 Getting API Keys

To use this app, you'll need API keys for both **Gemini** and **NVIDIA NIM**. Here's how to obtain them:

### 🌐 Gemini API Key
Gemini is Google's family of generative AI models. To get an API key:

1. Visit the [Google AI Studio](https://aistudio.google.com/api-keys).
2. Sign in with your Google account.
3. Click **"Create API Key"** and copy the key shown.
4. Use this key in your `.env` file or configuration as `GEMINI_API_KEY`.

> Note: Gemini API access may be limited based on region or account eligibility. Check the Gemini API [Rate Limits here](https://ai.google.dev/gemini-api/docs/rate-limits)

### 🚀 NVIDIA NIM API Key
NIM (NVIDIA Inference Microservices) provides hosted models via REST APIs. To get started:

1. Go to the [NVIDIA API Catalog](https://build.nvidia.com/?integrate_nim=true&hosted_api=true&modal=integrate-nim).
2. Choose a model (e.g., `nim-gemma`, `nim-mistral`, etc.) and click **"Get API Key"**.
3. Sign in or create an NVIDIA account if prompted.
4. Copy your key and use it as `NVIDIA_NIM_API_KEY` in your environment.

> Tip: You can test NIM endpoints directly in the browser before integrating.

---

Once you have both keys, store them securely and never commit them to version control.

---


### 🧪 How to Use
Upload a PDF — drag and drop your document

Click “📄 Process Document” — the app will split, embed, and store the content

Enter a query — ask a question like:

“What are the key findings?”

“Summarize the methodology.”

“What does the report say about climate change?”

Click “🔍 Ask a Question” — get semantic search results and an LLM-generated answer

---

### ⚙️ Configuration
All secrets are loaded from .env or Hugging Face Secrets tab:

| Variable        | Description                                 |
|------------------|-----------------------------------------|
| GOOGLE_API_KEY        | Gemini LLM API key  |
| NVIDIA_API_KEY         | NVIDIA LLM API key  |
| CHROMA_DIR  | Path to store Chroma vector DB                    |

---

### 🧩 Customization

Switch between NVIDIA and Gemini embeddings in process_pdf()

Change LLM model in search_query() (bytedance/seed-oss-36b-instruct, gemini-2.5-pro, etc.)

Tune chunk size and overlap in RecursiveCharacterTextSplitter

Add dropdowns to UI for model selection (optional)

---

### 📁 File Structure

```Code
semantic-search/
├── .env
├── .github/
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── app/
    ├── app.py
    └── config.py
```

---

## 📜 License

This project is open-source and distributed under the **[MIT License](https://opensource.org/licenses/MIT)**. Feel free to use, modify, and distribute it with attribution.

---


## 🤝 Acknowledgements

- [LangChain](https://www.langchain.com) — Powerful framework for orchestrating LLMs, embeddings, and retrieval pipelines.
- [ChromaDB](https://www.trychroma.com/) — Fast and flexible open-source vector database for semantic search.
- [NVIDIA AI Endpoints](https://build.nvidia.com/models) — Hosted LLM and embedding APIs including Seed OSS and NV-Embed.
- [Google Gemini](https://aistudio.google.com/welcome) — Robust multimodal LLM platform offering text embeddings and chat models.
- [Gradio](https://www.gradio.app) — Simple and elegant Python library for building machine learning interfaces.
- [PyMuPDF](https://pymupdf.readthedocs.io) — Lightweight PDF parser for fast and accurate text extraction.
- [Docker](https://www.docker.com) — Containerization platform for reproducible deployment across environments.
- [Hugging Face Spaces](https://huggingface.co/spaces) — Free hosting platform for ML demos with secret management and GPU support.
