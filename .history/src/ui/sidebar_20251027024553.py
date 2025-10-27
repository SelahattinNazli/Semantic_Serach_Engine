"""
Sidebar UI components.
"""

import streamlit as st
from typing import Optional

from src.utils.pdf_processor import PDFProcessor, save_uploaded_file, cleanup_temp_file
from src.models.vector_store import add_documents_to_store, create_retriever
from langchain_chroma import Chroma


def render_sidebar(
    pdf_processor: PDFProcessor,
    vector_store: Chroma,
    search_type: str,
    retrieval_k: int,
) -> None:
    """
    Render the sidebar with file upload and document info.

    Args:
        pdf_processor: PDFProcessor instance
        vector_store: ChromaDB instance
        search_type: Type of search for retriever
        retrieval_k: Number of documents to retrieve
    """
    with st.sidebar:
        st.markdown("### 🔍 Semantic Search Engine")
        st.markdown("---")

        # File Upload Section
        st.markdown("#### 📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"], help="Upload a PDF document to analyze"
        )

        # Process uploaded file
        if uploaded_file is not None:
            if st.session_state.pdf_name != uploaded_file.name:
                _process_uploaded_file(
                    uploaded_file, pdf_processor, vector_store, search_type, retrieval_k
                )

        st.markdown("---")

        # Document Info Section
        _render_document_info()

        st.markdown("---")

        # About Section
        _render_about_section()


def _process_uploaded_file(
    uploaded_file,
    pdf_processor: PDFProcessor,
    vector_store: Chroma,
    search_type: str,
    retrieval_k: int,
) -> None:
    """Process the uploaded PDF file."""
    with st.spinner("Processing document..."):
        temp_file_path = None
        try:
            # Save uploaded file
            temp_file_path = save_uploaded_file(uploaded_file, uploaded_file.name)

            # Process PDF
            chunks = pdf_processor.process_pdf(temp_file_path)

            # Add to vector store
            add_documents_to_store(vector_store, chunks)

            # Create retriever
            retriever = create_retriever(vector_store, search_type, retrieval_k)

            # Update session state
            st.session_state.retriever = retriever
            st.session_state.vector_store_ready = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []  # Clear previous chat

            st.success("✅ Document processed!")

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

        finally:
            if temp_file_path:
                cleanup_temp_file(temp_file_path)


def _render_document_info() -> None:
    """Render the current document information section."""
    if st.session_state.vector_store_ready:
        st.markdown("#### 📊 Current Document")
        st.info(f"**{st.session_state.pdf_name}**")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("No document loaded")


def _render_about_section() -> None:
    """Render the about section."""
    st.markdown("#### ℹ️ About")
    st.markdown("""
    This tool uses:
    - **Qwen 3** for intelligent responses
    - **Sentence Transformers** for embeddings
    - **ChromaDB** for vector storage
    
    Ask questions about your uploaded PDF!
    """)
