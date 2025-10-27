"""
Embedding model for document vectorization.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st


class SentenceTransformerEmbeddings:
    """
    Custom embeddings class using SentenceTransformers.
    Compatible with LangChain's embedding interface.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        return SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


@st.cache_resource
def load_embedding_model(
    model_name: str, device: str = "cpu"
) -> SentenceTransformerEmbeddings:
    """
    Load and cache the embedding model.

    Args:
        model_name: Name of the model to load
        device: Device to use for inference

    Returns:
        Initialized embedding model
    """
    return SentenceTransformerEmbeddings(model_name=model_name, device=device)
