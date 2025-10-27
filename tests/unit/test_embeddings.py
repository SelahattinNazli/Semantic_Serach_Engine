"""
Unit tests for embeddings module.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.models.embeddings import SentenceTransformerEmbeddings


class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformerEmbeddings class."""

    @patch("src.models.embeddings.SentenceTransformer")
    def test_initialization(self, mock_transformer):
        """Test embedding model initialization."""
        embeddings = SentenceTransformerEmbeddings(
            model_name="test-model", device="cpu"
        )

        assert embeddings.model_name == "test-model"
        assert embeddings.device == "cpu"
        mock_transformer.assert_called_once_with("test-model", device="cpu")

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embed_documents(self, mock_transformer):
        """Test embedding multiple documents."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings()

        # Test
        texts = ["document 1", "document 2"]
        result = embeddings.embed_documents(texts)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 3
        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embed_query(self, mock_transformer):
        """Test embedding a single query."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings()

        # Test
        query = "test query"
        result = embeddings.embed_query(query)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 3
        mock_model.encode.assert_called_once_with([query], convert_to_numpy=True)

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embed_empty_documents(self, mock_transformer):
        """Test embedding empty list of documents."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([])
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings()

        result = embeddings.embed_documents([])

        assert isinstance(result, list)
        mock_model.encode.assert_called_once()

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embed_documents_returns_correct_shape(self, mock_transformer):
        """Test that embeddings have correct shape."""
        mock_model = MagicMock()
        # Simulate 384-dimensional embeddings (common for MiniLM)
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings()

        texts = ["doc1", "doc2", "doc3"]
        result = embeddings.embed_documents(texts)

        assert len(result) == 3
        assert len(result[0]) == 384

    @patch("src.models.embeddings.SentenceTransformer")
    def test_embed_query_single_dimension(self, mock_transformer):
        """Test that query embedding returns 1D list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_transformer.return_value = mock_model

        embeddings = SentenceTransformerEmbeddings()

        result = embeddings.embed_query("test")

        # Should be 1D list (not nested)
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    @patch("src.models.embeddings.SentenceTransformer")
    def test_different_devices(self, mock_transformer):
        """Test initialization with different devices."""
        # Test CPU
        embeddings_cpu = SentenceTransformerEmbeddings(device="cpu")
        assert embeddings_cpu.device == "cpu"

        # Test CUDA
        embeddings_cuda = SentenceTransformerEmbeddings(device="cuda")
        assert embeddings_cuda.device == "cuda"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    @patch("src.models.embeddings.SentenceTransformer")
    def test_load_embedding_model_caching(self, mock_transformer):
        """Test that model loading is cached."""
        from src.models.embeddings import load_embedding_model

        # Clear any existing cache
        load_embedding_model.clear()

        # Load model twice
        model1 = load_embedding_model("test-model", "cpu")
        model2 = load_embedding_model("test-model", "cpu")

        # Should return the same instance (cached)
        assert model1 is model2

    @patch("src.models.embeddings.SentenceTransformer")
    def test_load_with_different_params(self, mock_transformer):
        """Test loading with different parameters."""
        from src.models.embeddings import load_embedding_model

        load_embedding_model.clear()

        model1 = load_embedding_model("model1", "cpu")
        model2 = load_embedding_model("model2", "cpu")

        # Different parameters should return different instances
        assert model1 is not model2
