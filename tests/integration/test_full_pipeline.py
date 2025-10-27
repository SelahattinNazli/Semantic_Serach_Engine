"""
Integration tests for the complete semantic search pipeline.
Tests the interaction between all components working together.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
import numpy as np

from src.utils.config import ModelConfig, ProcessingConfig
from src.utils.pdf_processor import PDFProcessor
from src.models.embeddings import SentenceTransformerEmbeddings
from src.models.vector_store import (
    initialize_vector_store,
    add_documents_to_store,
    create_retriever,
    retrieve_documents,
)
from src.models.llm import generate_response


class TestFullPipeline:
    """Integration tests for the complete document processing pipeline."""

    @patch("src.models.embeddings.SentenceTransformer")
    @patch("src.models.vector_store.Chroma")
    def test_document_to_retrieval_pipeline(
        self, mock_chroma, mock_transformer, sample_documents, temp_dir
    ):
        """Test complete pipeline from documents to retrieval."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])  # numpy array
        mock_transformer.return_value = mock_model

        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = sample_documents[:1]
        mock_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_store

        # 1. Process documents
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.split_documents(sample_documents)

        assert len(chunks) > 0

        # 2. Create embeddings
        embedding_model = SentenceTransformerEmbeddings()

        # 3. Initialize vector store
        vector_store = initialize_vector_store(
            _embedding_model=embedding_model,
            collection_name="test_integration",
            persist_directory=str(temp_dir / "db"),
        )

        # 4. Add documents
        add_documents_to_store(vector_store, chunks)
        mock_store.add_documents.assert_called_once()

        # 5. Create retriever
        retriever = create_retriever(vector_store, search_type="similarity", k=3)

        # 6. Retrieve documents
        results = retrieve_documents(retriever, "test query")

        assert len(results) > 0
        assert isinstance(results[0], Document)

    @patch("src.models.embeddings.SentenceTransformer")
    @patch("src.models.vector_store.Chroma")
    def test_end_to_end_with_config(
        self, mock_chroma, mock_transformer, sample_documents, temp_dir
    ):
        """Test end-to-end pipeline using configuration objects."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])  # numpy array
        mock_transformer.return_value = mock_model

        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = sample_documents
        mock_store.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_store

        # Use actual config objects
        model_config = ModelConfig()
        processing_config = ProcessingConfig()

        # Initialize components with config
        processor = PDFProcessor(
            chunk_size=processing_config.chunk_size,
            chunk_overlap=processing_config.chunk_overlap,
        )

        embedding_model = SentenceTransformerEmbeddings(
            model_name=model_config.embedding_model_name,
            device=model_config.embedding_device,
        )

        vector_store = initialize_vector_store(
            _embedding_model=embedding_model,
            collection_name=model_config.collection_name,
            persist_directory=str(temp_dir / "db"),
        )

        # Process pipeline
        chunks = processor.split_documents(sample_documents)
        add_documents_to_store(vector_store, chunks)
        retriever = create_retriever(
            vector_store,
            search_type=processing_config.search_type,
            k=processing_config.retrieval_k,
        )
        results = retrieve_documents(retriever, "What is AI?")

        assert len(results) > 0

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embedding_and_chunking_integration(
        self, mock_transformer, sample_documents
    ):
        """Test that embeddings work correctly with chunked documents."""
        # Setup
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )  # numpy array
        mock_transformer.return_value = mock_model

        # Process
        processor = PDFProcessor(chunk_size=50, chunk_overlap=10)
        chunks = processor.split_documents(sample_documents)

        embedding_model = SentenceTransformerEmbeddings()

        # Get embeddings for chunks
        texts = [chunk.page_content for chunk in chunks[:2]]
        embeddings = embedding_model.embed_documents(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3

    # diğer testler aynı şekilde kalabilir
