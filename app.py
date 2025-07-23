import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY", None))

if not api_key:
    st.error("🚨 Please set your OPENROUTER_API_KEY in .env or .streamlit/secrets.toml")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL = "deepseek/deepseek-r1:free"

# --- Extract text from PDF ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Split text into overlapping chunks ---
def chunk_text(text, chunk_size=800, overlap=200):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

# --- Embed & Index Chunks ---
@st.cache_resource(show_spinner="🔍 Indexing PDF...")
def create_index(chunks):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
    embeddings = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, embedder

# --- Find Relevant Chunks for Query ---
def search_similar_chunks(query, chunks, index, embedder, k=4):
    q_embed = embedder.encode([query]).astype("float32")
    _, I = index.search(q_embed, k)
    return [chunks[i] for i in I[0]]

# --- Streamlit UI ---
st.set_page_config(page_title="PDF QA Chatbot", layout="centered")
st.title("📄🤖 PDF QA Chatbot")
st.caption("Powered by DeepSeek-R1 (Free via OpenRouter)")

uploaded_pdf = st.file_uploader("📎 Upload a PDF", type=["pdf"])

if uploaded_pdf:
    text = extract_text_from_pdf(uploaded_pdf)
    chunks = list(chunk_text(text))
    index, embeddings, embedder = create_index(chunks)
    st.success(f"✅ Loaded and indexed {len(chunks)} chunks from PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).markdown(msg)

    prompt = st.chat_input("Ask a question about the PDF")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(("user", prompt))

        context_chunks = search_similar_chunks(prompt, chunks, index, embedder)
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a helpful assistant. Use only the context below to answer the user's question. "
            "If the answer isn't there, say you don't know.\n\n"
            f"Context:\n{context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        with st.spinner("🤖 Thinking…"):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://github.com/yourusername/pdf-qa-chatbot",
                    "X-Title": "PDF QA Chatbot",
                }
            )

            answer_box = st.empty()
            final_answer = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                final_answer += delta
                answer_box.markdown(final_answer + "▌")
            answer_box.markdown(final_answer)

            st.session_state.chat_history.append(("assistant", final_answer))
