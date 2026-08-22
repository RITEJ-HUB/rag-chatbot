import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

st.set_page_config(page_title="RAG Chatbot — Chat with your documents", page_icon="💬", layout="wide")

st.title("💬 Chat with your documents")
st.caption("Upload a PDF or text file and ask questions about it — powered by RAG (Retrieval-Augmented Generation)")

# --- Sidebar: file upload ---
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or .txt/.md file", type=["pdf", "txt", "md"])
    st.divider()
    st.caption("How it works:\n1. Your document is split into chunks\n2. Each chunk is converted to an embedding\n3. Your question searches for the most relevant chunks\n4. The LLM answers using only those chunks")

# --- Initialize session state ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

# --- Process uploaded file ---
if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:
    with st.spinner("Processing document... (chunking + embedding)"):
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_upload" + os.path.splitext(uploaded_file.name)[1])
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.processed_filename = uploaded_file.name
        st.session_state.messages = []

        os.remove(temp_path)

    st.success(f"✅ Processed '{uploaded_file.name}' — {len(chunks)} chunks indexed. Ask a question below!")

# --- Chat interface ---
if st.session_state.vectorstore is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                # Step 1: retrieve the most relevant chunks (manual, version-proof)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Step 2: build the prompt with retrieved context
                prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

                # Step 3: call the LLM
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)

                with st.expander("📄 Source chunks used"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF or text document in the sidebar to get started.")
