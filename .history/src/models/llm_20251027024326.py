"""
Large Language Model for response generation.
"""

from typing import Iterator
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


@st.cache_resource
def load_llm(model_name: str, temperature: float = 0.2) -> ChatOllama:
    """
    Load and cache the LLM model.

    Args:
        model_name: Name of the Ollama model
        temperature: Sampling temperature

    Returns:
        Initialized ChatOllama instance
    """
    return ChatOllama(model=model_name, temperature=temperature)


def generate_response(
    llm: ChatOllama,
    query: str,
    retrieved_docs: list[Document],
    system_prompt_template: str,
) -> Iterator[str]:
    """
    Generate a streaming response from the LLM.

    Args:
        llm: The language model instance
        query: User's question
        retrieved_docs: Documents retrieved from vector store
        system_prompt_template: Template for the system prompt

    Yields:
        Chunks of the generated response
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt_template)]
    )

    final_prompt = prompt_template.invoke(
        {"question": query, "document": retrieved_docs}
    )

    for chunk in llm.stream(final_prompt):
        yield chunk.content


def get_full_response(
    llm: ChatOllama,
    query: str,
    retrieved_docs: list[Document],
    system_prompt_template: str,
) -> str:
    """
    Generate a complete (non-streaming) response from the LLM.

    Args:
        llm: The language model instance
        query: User's question
        retrieved_docs: Documents retrieved from vector store
        system_prompt_template: Template for the system prompt

    Returns:
        Complete response as a string
    """
    response = ""
    for chunk in generate_response(llm, query, retrieved_docs, system_prompt_template):
        response += chunk
    return response
