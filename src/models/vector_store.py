"""
Vector store management for document embeddings.
"""

from typing import List
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.models.embeddings import SentenceTransformerEmbeddings


@st.cache_resource
def initialize_vector_store(
    _embedding_model: SentenceTransformerEmbeddings,
    collection_name: str,
    persist_directory: str,
) -> Chroma:
    """
    Initialize and cache the vector store.

    Args:
        _embedding_model: Embedding model instance (prefixed with _ to avoid hashing)
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the database

    Returns:
        Initialized Chroma vector store
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=_embedding_model,
        persist_directory=persist_directory,
    )


def add_documents_to_store(vector_store: Chroma, documents: List[Document]) -> None:
    """
    Add documents to the vector store.

    Args:
        vector_store: The ChromaDB instance
        documents: List of documents to add
    """
    vector_store.add_documents(documents=documents)


def create_retriever(
    vector_store: Chroma, search_type: str = "similarity", k: int = 5
) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store.

    Args:
        vector_store: The ChromaDB instance
        search_type: Type of search ('similarity' or 'mmr')
        k: Number of documents to retrieve

    Returns:
        Configured retriever
    """
    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})


def retrieve_documents(retriever: VectorStoreRetriever, query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query.

    Args:
        retriever: The retriever instance
        query: User's query

    Returns:
        List of relevant documents
    """
    return retriever.invoke(query)
