"""
Unit tests for vector store module.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.models.vector_store import (
    add_documents_to_store,
    create_retriever,
    retrieve_documents,
)


class TestAddDocumentsToStore:
    """Tests for add_documents_to_store function."""

    def test_add_documents(self, sample_documents):
        """Test adding documents to vector store."""
        mock_store = MagicMock()

        add_documents_to_store(mock_store, sample_documents)

        mock_store.add_documents.assert_called_once_with(documents=sample_documents)

    def test_add_empty_documents(self):
        """Test adding empty list of documents."""
        mock_store = MagicMock()

        add_documents_to_store(mock_store, [])

        mock_store.add_documents.assert_called_once_with(documents=[])

    def test_add_documents_with_metadata(self):
        """Test adding documents with metadata."""
        mock_store = MagicMock()
        docs = [
            Document(page_content="test", metadata={"source": "test.pdf", "page": 1})
        ]

        add_documents_to_store(mock_store, docs)

        called_docs = mock_store.add_documents.call_args[1]["documents"]
        assert called_docs[0].metadata["source"] == "test.pdf"


class TestCreateRetriever:
    """Tests for create_retriever function."""

    def test_create_retriever_default_params(self):
        """Test creating retriever with default parameters."""
        mock_store = MagicMock()

        retriever = create_retriever(mock_store)

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def test_create_retriever_custom_params(self):
        """Test creating retriever with custom parameters."""
        mock_store = MagicMock()

        retriever = create_retriever(mock_store, search_type="mmr", k=3)

        mock_store.as_retriever.assert_called_once_with(
            search_type="mmr", search_kwargs={"k": 3}
        )

    def test_create_retriever_similarity_search(self):
        """Test retriever with similarity search."""
        mock_store = MagicMock()

        create_retriever(mock_store, search_type="similarity", k=10)

        call_args = mock_store.as_retriever.call_args
        assert call_args[1]["search_type"] == "similarity"
        assert call_args[1]["search_kwargs"]["k"] == 10

    def test_create_retriever_mmr_search(self):
        """Test retriever with MMR search."""
        mock_store = MagicMock()

        create_retriever(mock_store, search_type="mmr", k=7)

        call_args = mock_store.as_retriever.call_args
        assert call_args[1]["search_type"] == "mmr"
        assert call_args[1]["search_kwargs"]["k"] == 7


class TestRetrieveDocuments:
    """Tests for retrieve_documents function."""

    def test_retrieve_documents(self, sample_documents):
        """Test retrieving documents."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = sample_documents[:2]

        query = "test query"
        results = retrieve_documents(mock_retriever, query)

        mock_retriever.invoke.assert_called_once_with(query)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieve_documents_empty_results(self):
        """Test retrieving documents with no results."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        results = retrieve_documents(mock_retriever, "no match query")

        assert results == []

    def test_retrieve_documents_with_different_queries(self):
        """Test retrieval with different query types."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="result", metadata={})
        ]

        queries = [
            "simple query",
            "What is AI?",
            "Explain machine learning in detail",
            "",
        ]

        for query in queries:
            results = retrieve_documents(mock_retriever, query)
            mock_retriever.invoke.assert_called_with(query)
            assert isinstance(results, list)

    def test_retrieve_documents_preserves_metadata(self):
        """Test that retrieval preserves document metadata."""
        mock_retriever = MagicMock()
        docs_with_metadata = [
            Document(
                page_content="content",
                metadata={"source": "file.pdf", "page": 1, "score": 0.95},
            )
        ]
        mock_retriever.invoke.return_value = docs_with_metadata

        results = retrieve_documents(mock_retriever, "query")

        assert results[0].metadata["source"] == "file.pdf"
        assert results[0].metadata["page"] == 1
        assert results[0].metadata["score"] == 0.95


class TestInitializeVectorStore:
    """Tests for initialize_vector_store function."""

    @patch("src.models.vector_store.Chroma")
    def test_initialize_vector_store(self, mock_chroma, mock_embedding_model):
        """Test initializing vector store."""
        from src.models.vector_store import initialize_vector_store

        # Clear cache
        initialize_vector_store.clear()

        store = initialize_vector_store(
            _embedding_model=mock_embedding_model,
            collection_name="test_collection",
            persist_directory="./test_db",
        )

        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embedding_model,
            persist_directory="./test_db",
        )

    @patch("src.models.vector_store.Chroma")
    def test_initialize_vector_store_caching(self, mock_chroma, mock_embedding_model):
        """Test that vector store initialization is cached."""
        from src.models.vector_store import initialize_vector_store

        initialize_vector_store.clear()

        store1 = initialize_vector_store(mock_embedding_model, "test", "./db")
        store2 = initialize_vector_store(mock_embedding_model, "test", "./db")

        # Should only call Chroma once due to caching
        assert mock_chroma.call_count == 1
