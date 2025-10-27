"""
Chat interface components.
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import VectorStoreRetriever

from src.models.llm import generate_response
from src.models.vector_store import retrieve_documents


def render_chat_interface(llm: ChatOllama, system_prompt_template: str) -> None:
    """
    Render the main chat interface.

    Args:
        llm: The language model instance
        system_prompt_template: Template for system prompts
    """
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        _handle_user_input(prompt, llm, system_prompt_template)


def _handle_user_input(
    prompt: str, llm: ChatOllama, system_prompt_template: str
) -> None:
    """
    Handle user input and generate response.

    Args:
        prompt: User's question
        llm: The language model instance
        system_prompt_template: Template for system prompts
    """
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.retriever
                docs_retrieved = retrieve_documents(retriever, prompt)

                # Generate streaming response
                response_placeholder = st.empty()
                full_response = ""

                for chunk in generate_response(
                    llm, prompt, docs_retrieved, system_prompt_template
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # Add to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


def render_welcome_screen() -> None:
    """Render the welcome screen when no document is loaded."""
    from src.ui.styles import WELCOME_HTML, get_feature_cards_html

    st.markdown(WELCOME_HTML, unsafe_allow_html=True)
    st.markdown(get_feature_cards_html(), unsafe_allow_html=True)
