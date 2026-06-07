"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Model Configuration
    LLM_MODEL = "groq:llama-3.1-8b-instant"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_groq_api_key(cls):
        """Get Groq API key from Streamlit secrets or environment."""
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
            if key:
                return key
        except Exception:
            pass

        return os.getenv("GROQ_API_KEY")

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model."""
        groq_api_key = cls.get_groq_api_key()

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Add it in Streamlit Cloud Secrets.")

        os.environ["GROQ_API_KEY"] = groq_api_key
        return init_chat_model(cls.LLM_MODEL)