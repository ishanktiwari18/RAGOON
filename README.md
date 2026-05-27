# 🦝 RAGOON — Agentic RAG Document Search System

RAGOON is an Agentic Retrieval-Augmented Generation (RAG) system built using:

- LangGraph
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLMs
- Streamlit

The project enables intelligent document retrieval and reasoning using autonomous ReAct-style workflows.

---

# 🚀 Features

- Agentic RAG pipeline using LangGraph
- ReAct-style retrieval workflow
- FAISS vector database
- HuggingFace embedding models
- Groq-powered LLM inference
- Streamlit UI
- Multi-document retrieval
- Source document inspection
- Modular architecture
- Extensible tool system

---

# 🧠 Architecture

```text
User Query
    ↓
LangGraph Workflow
    ↓
Retriever Tool
    ↓
FAISS Vector Search
    ↓
HuggingFace Embeddings
    ↓
Groq LLM Reasoning
    ↓
Final Response
```

---

# 📂 Project Structure

```text
RAGOON/
│
├── data/
│   ├── attention.pdf
│   └── url.txt
│
├── src/
│   ├── config/
│   ├── document_ingestion/
│   ├── graph_builder/
│   ├── node/
│   ├── state/
│   └── vectorstore/
│
├── .env
├── requirements.txt
├── streamlit_app.py
├── main.py
└── README.md
```

---

# ⚙️ Tech Stack

## Frameworks

- LangChain
- LangGraph
- Streamlit

## Vector Database

- FAISS

## Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

## LLM Provider

- Groq

## Model

- llama-3.3-70b-versatile

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/RAGOON.git

cd RAGOON
```

---

## 2. Create Virtual Environment

### Using UV

```bash
uv venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Run Application

```bash
python -m streamlit run streamlit_app.py
```

---

# 🌐 Default Data Sources

The system currently indexes:

- https://lilianweng.github.io/posts/2023-06-23-agent/
- https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

---

# 🧪 Example Queries

## Agentic AI

- What is task decomposition?
- Explain ReAct prompting.
- Compare ReAct and Chain-of-Thought prompting.
- How does memory work in autonomous agents?

## Diffusion Models

- What are diffusion transformers?
- Explain temporal consistency in video generation.
- Compare video diffusion and image diffusion.

---

# 🔄 RAG Workflow

## 1. Document Ingestion

Documents are loaded and chunked.

## 2. Embedding Generation

Chunks are converted into vector embeddings using HuggingFace models.

## 3. Vector Storage

Embeddings are stored in FAISS.

## 4. Retrieval

Relevant chunks are retrieved using similarity search.

## 5. Agentic Reasoning

LangGraph + Groq LLM reason over retrieved context.

## 6. Final Answer

Grounded response is returned to the user.

---



# 🛠 Current Limitations

- Wikipedia tool calling with Groq may produce malformed function calls
- FAISS currently runs in-memory
- No persistent conversation memory
- Retrieval reranking not implemented yet

---

# 📚 Core Concepts Used

- Retrieval-Augmented Generation (RAG)
- ReAct Agents
- Vector Similarity Search
- Embedding Models
- Autonomous Reasoning
- Tool-Augmented LLMs

---

