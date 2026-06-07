"""Streamlit UI for Agentic RAG System - Redesigned"""

import os
os.environ["USER_AGENT"] = "RAGOON/1.0"

import uuid
import builtins
builtins.uuid = uuid

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


st.set_page_config(
    page_title="RAGOON",
    page_icon="🦝",
    layout="centered"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #ECEAE4;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stCacheSpinner"],
[data-testid="stStatusWidget"],
.stCacheSpinner,
div[class*="StatusWidget"],
div[class*="cacheSpinner"],
iframe[title="st_cache_spinner"] {
    display: none !important;
}

.block-container {
    max-width: 800px !important;
    padding: 3.5rem 2rem 5rem 2rem !important;
}

.hero-block {
    background: linear-gradient(135deg, #5B50D6 0%, #7B6FEF 100%);
    border-radius: 28px;
    padding: 48px 44px 44px 44px;
    margin-bottom: 32px;
    box-shadow: 0 12px 40px rgba(91, 80, 214, 0.28);
}

.hero-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.50);
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 62px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.0;
    margin-bottom: 16px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 14.5px;
    font-weight: 300;
    color: rgba(255,255,255,0.62);
    line-height: 1.7;
    max-width: 440px;
}

.pills-row {
    display: flex;
    gap: 10px;
    margin-bottom: 26px;
    flex-wrap: wrap;
    align-items: center;
}

.pill {
    background: #FFFFFF;
    border: 2px solid #DDD8F5;
    border-radius: 100px;
    padding: 8px 22px;
    font-size: 13px;
    font-weight: 500;
    color: #5B50D6;
    white-space: nowrap;
    line-height: 1;
}

.pill-active {
    background: #5B50D6;
    border-color: #5B50D6;
    color: #FFFFFF;
    font-weight: 600;
}

.ready-banner {
    background: #EDEAFB;
    border: 2px solid #C8C1F0;
    border-radius: 16px;
    padding: 16px 22px;
    margin-bottom: 24px;
    font-size: 13.5px;
    font-weight: 500;
    color: #4A42A8;
    letter-spacing: 0.1px;
}

div[data-testid="stAlert"] {
    display: none !important;
}

.stTextInput > div > div > input {
    background-color: #FFFFFF !important;
    border: 2px solid #D8D3F0 !important;
    border-radius: 16px !important;
    padding: 17px 22px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    color: #1A1630 !important;
    caret-color: #1A1630 !important;
    box-shadow: 0 2px 12px rgba(91,80,214,0.07) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #B8B2D8 !important;
    font-weight: 300 !important;
}

.stTextInput > div > div > input:focus {
    border-color: #5B50D6 !important;
    box-shadow: 0 0 0 3px rgba(91,80,214,0.12) !important;
    outline: none !important;
}

.stTextInput label {
    display: none !important;
}

.stTextInput > div {
    margin-bottom: 14px;
}

.stFormSubmitButton > button {
    background: linear-gradient(135deg, #5B50D6 0%, #7B6FEF 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 15px 48px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    width: auto !important;
    min-width: 160px !important;
    box-shadow: 0 6px 20px rgba(91,80,214,0.30) !important;
}

.stFormSubmitButton > button:focus {
    box-shadow: 0 6px 20px rgba(91,80,214,0.30) !important;
    outline: none !important;
}

.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #9990C8;
    margin-bottom: 14px;
    margin-top: 36px;
}

.answer-card {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 34px 36px;
    border: 2px solid #E8E4FA;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(91,80,214,0.08);
}

.answer-q {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #12102A;
    margin-bottom: 16px;
    line-height: 1.35;
}

.answer-body {
    font-size: 14.5px;
    font-weight: 300;
    color: #4E4870;
    line-height: 1.9;
    white-space: pre-wrap;
}

.time-badge {
    display: inline-block;
    background: #EDEAFB;
    border: 1.5px solid #C8C1F0;
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 11.5px;
    font-weight: 600;
    color: #5B50D6;
    margin-top: 18px;
    letter-spacing: 0.5px;
}

details {
    background: #1C1830 !important;
    border-radius: 16px !important;
    border: none !important;
    overflow: hidden !important;
    margin-bottom: 16px !important;
}

details > summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #A89CF5 !important;
    padding: 18px 24px !important;
    letter-spacing: 0.3px !important;
    list-style: none !important;
    cursor: pointer !important;
}

details > summary::-webkit-details-marker {
    display: none;
}

.streamlit-expanderContent {
    padding: 4px 4px 16px 4px !important;
    background: transparent !important;
}

.source-card {
    background: #2A2545;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 0 12px 10px 12px;
    border: 1px solid #3D3660;
}

.source-num {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7B6FEF;
    margin-bottom: 8px;
}

.source-text {
    font-size: 13px;
    font-weight: 300;
    color: #B8B0D8;
    line-height: 1.72;
}

.divider {
    border: none;
    border-top: 2px solid #DEDAD2;
    margin: 36px 0 0 0;
}

.history-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 22px 26px;
    border: 2px solid #E8E4FA;
    margin-bottom: 12px;
    box-shadow: 0 2px 14px rgba(91,80,214,0.06);
}

.history-q {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 700;
    color: #12102A;
    margin-bottom: 7px;
    line-height: 1.4;
}

.history-a {
    font-size: 13px;
    font-weight: 300;
    color: #6A6490;
    line-height: 1.68;
    margin-bottom: 10px;
}

.history-meta {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #B8B2D8;
}

.stSpinner > div > div {
    border-top-color: #5B50D6 !important;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "rag_system": None,
        "initialized": False,
        "history": [],
        "last_result": None,
        "ready_msg": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm()

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        vector_store = VectorStore()

        documents = doc_processor.process_urls(Config.DEFAULT_URLS)
        vector_store.create_vectorstore(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
        )

        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        return None, str(e)


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_answer(question: str, answer: str, elapsed: float):
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-q">{esc(question)}</div>
        <div class="answer-body">{esc(answer)}</div>
        <div class="time-badge">{elapsed:.2f} s</div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(docs):
    with st.expander("Source Documents"):
        for index, doc in enumerate(docs, 1):
            text = esc(doc.page_content[:320])

            st.markdown(f"""
            <div class="source-card">
                <div class="source-num">Source {index}</div>
                <div class="source-text">{text}...</div>
            </div>
            """, unsafe_allow_html=True)


def render_history(history):
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recent Searches</div>', unsafe_allow_html=True)

    for item in reversed(history[-3:]):
        st.markdown(f"""
        <div class="history-card">
            <div class="history-q">{esc(item["question"])}</div>
            <div class="history-a">{esc(item["answer"][:200])}...</div>
            <div class="history-meta">{item["time"]:.2f} s</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    init_session_state()

    st.markdown("""
    <div class="hero-block">
        <div class="hero-label">Agentic Research</div>
        <div class="hero-title">🦝Ragoon</div>
        <div class="hero-subtitle">
            Interact with cutting-edge research on AI Agents and Diffusion Video Models
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pills-row">
        <span class="pill pill-active">All</span>
        <span class="pill">AI Agents</span>
        <span class="pill">Diffusion Models</span>
        <span class="pill">Research</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            rag_system, result = initialize_rag()

            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.session_state.ready_msg = (
                    f"System ready — {result} document chunks loaded"
                )
            else:
                st.session_state.ready_msg = f"Initialization error: {result}"

    if st.session_state.ready_msg:
        st.markdown(
            f'<div class="ready-banner">{esc(st.session_state.ready_msg)}</div>',
            unsafe_allow_html=True,
        )

    with st.form("search_form"):
        question = st.text_input(
            "Question",
            placeholder="Search across research papers...",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Search")

    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed = time.time() - start_time

            entry = {
                "question": question,
                "answer": result["answer"],
                "docs": result["retrieved_docs"],
                "time": elapsed,
            }

            st.session_state.history.append(entry)
            st.session_state.last_result = entry

    if st.session_state.last_result:
        result = st.session_state.last_result
        render_answer(result["question"], result["answer"], result["time"])
        render_sources(result["docs"])

    if st.session_state.history:
        render_history(st.session_state.history)


if __name__ == "__main__":
    main()