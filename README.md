# 💬 RAG Chatbot — Chat with Your Documents

An interactive chatbot that answers questions grounded in your own documents (PDF or text), using Retrieval-Augmented Generation (RAG) instead of relying purely on an LLM's training data.

**🔗 Live demo: [rag-chatbot-4nifbevynuokzfdzey2wv8.streamlit.app](https://rag-chatbot-4nifbevynuokzfdzey2wv8.streamlit.app)**

## Overview

Standard LLMs can't answer questions about documents they've never seen. This project solves that with RAG: your document is split into chunks, each chunk is converted into an embedding (a numeric representation of its meaning), and stored in a local vector database. When you ask a question, the app searches for the most relevant chunks and passes them to the LLM as context — so the answer is grounded in your actual document, with the source chunks shown for transparency.

## How it works

1. **Upload** — a PDF or text/markdown file via the sidebar.
2. **Chunking** — the document is split into ~500-character chunks with overlap, using LangChain's text splitter.
3. **Embedding** — each chunk is converted into a vector using a free, local embedding model (`sentence-transformers/all-MiniLM-L6-v2`) — no API cost for this step.
4. **Vector search** — chunks are stored in a FAISS vector index. On each question, the top 3 most relevant chunks are retrieved.
5. **Answer generation** — the retrieved chunks + the question are passed to an LLM (Groq's `openai/gpt-oss-20b`) to generate a grounded answer.
6. **Transparency** — every answer includes an expandable "Source chunks used" section, showing exactly what the model based its answer on.

## Tech Stack

- **Python**
- **LangChain** — document loading, chunking, orchestration
- **sentence-transformers** — local embeddings (`all-MiniLM-L6-v2`)
- **FAISS** — vector similarity search
- **Groq API** — fast, free-tier LLM inference (`openai/gpt-oss-20b`)
- **Streamlit** — chat interface
- **python-dotenv** — API key management

## Project Structure

```
rag-chatbot/
├── app.py           # Streamlit RAG chatbot
├── README.md
└── .gitignore
```

> Note: `.env` (containing the Groq API key) is excluded from this repo for security — see setup below.

## How to Run

1. Install dependencies:
   ```
   pip install streamlit langchain langchain-community langchain-groq sentence-transformers faiss-cpu python-dotenv pypdf
   ```
2. Get a free API key from [Groq](https://console.groq.com/keys).
3. Create a `.env` file in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Launch the app:
   ```
   streamlit run app.py
   ```
5. Upload a PDF or text file and start asking questions.

## Author

Ritej — [GitHub](https://github.com/RITEJ-HUB)
