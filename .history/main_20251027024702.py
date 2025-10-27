"""
Semantic Search Engine - Main Application Entry Point
"""

import os
import streamlit as st

# Remove HuggingFace tokens
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from src.utils.config import (
    model_config,
    processing_config,
    ui_config,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.models.embeddings import load_embedding_model
from src.models.llm import load_llm
from src.models.vector_store import initialize_vector_store
from src.utils.pdf_processor import PDFProcessor
from src.ui.sidebar import render_sidebar
from src.ui.chat import render_chat_interface, render_welcome_screen
from src.ui.styles import CUSTOM_CSS


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=ui_config.page_title,
        page_icon=ui_config.page_icon,
        layout=ui_config.layout,
        initial_sidebar_state=ui_config.sidebar_state,
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    """Main application logic."""
    # Configure page
    configure_page()

    # Initialize session state
    initialize_session_state()

    # Load models
    embedding_model = load_embedding_model(
        model_name=model_config.embedding_model_name,
        device=model_config.embedding_device,
    )

    vector_store = initialize_vector_store(
        _embedding_model=embedding_model,
        collection_name=model_config.collection_name,
        persist_directory=model_config.persist_directory,
    )

    llm = load_llm(
        model_name=model_config.llm_model_name, temperature=model_config.llm_temperature
    )

    # Initialize PDF processor
    pdf_processor = PDFProcessor(
        chunk_size=processing_config.chunk_size,
        chunk_overlap=processing_config.chunk_overlap,
        add_start_index=processing_config.add_start_index,
    )

    # Render sidebar
    render_sidebar(
        pdf_processor=pdf_processor,
        vector_store=vector_store,
        search_type=processing_config.search_type,
        retrieval_k=processing_config.retrieval_k,
    )

    # Render main content
    st.markdown(
        '<h1 class="main-header">AI-Powered Document Q&A</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload a PDF and ask questions using advanced semantic search</p>',
        unsafe_allow_html=True,
    )

    # Render chat interface or welcome screen
    if st.session_state.vector_store_ready:
        render_chat_interface(llm, SYSTEM_PROMPT_TEMPLATE)
    else:
        render_welcome_screen()


if __name__ == "__main__":
    main()
