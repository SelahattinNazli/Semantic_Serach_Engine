"""
Unit tests for PDF processor module.
"""

import os
from pathlib import Path
import pytest
from langchain_core.documents import Document

from src.utils.pdf_processor import PDFProcessor, save_uploaded_file, cleanup_temp_file


class TestPDFProcessor:
    """Tests for PDFProcessor class."""

    def test_initialization_default_values(self):
        """Test processor initialization with default values."""
        processor = PDFProcessor()

        assert processor.text_splitter is not None
        assert processor.text_splitter._chunk_size == 1000
        assert processor.text_splitter._chunk_overlap == 200

    def test_initialization_custom_values(self):
        """Test processor initialization with custom values."""
        processor = PDFProcessor(
            chunk_size=500, chunk_overlap=100, add_start_index=False
        )

        assert processor.text_splitter._chunk_size == 500
        assert processor.text_splitter._chunk_overlap == 100

    def test_split_documents(self, sample_documents):
        """Test document splitting functionality."""
        processor = PDFProcessor(chunk_size=50, chunk_overlap=10)

        # Split the documents
        chunks = processor.split_documents(sample_documents)

        # Verify we get chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_documents_preserves_metadata(self, sample_documents):
        """Test that splitting preserves document metadata."""
        processor = PDFProcessor()
        chunks = processor.split_documents(sample_documents)

        # Check that metadata is preserved
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_split_empty_documents(self):
        """Test splitting empty document list."""
        processor = PDFProcessor()
        chunks = processor.split_documents([])

        assert chunks == []

    def test_split_documents_with_long_text(self):
        """Test splitting documents with long text."""
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)

        long_text = "This is a test sentence. " * 50
        docs = [Document(page_content=long_text, metadata={"source": "test"})]

        chunks = processor.split_documents(docs)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should be roughly the chunk_size
        for chunk in chunks:
            assert len(chunk.page_content) <= processor.text_splitter._chunk_size + 100


class TestSaveUploadedFile:
    """Tests for save_uploaded_file function."""

    def test_save_uploaded_file(self, mocker, temp_dir):
        """Test saving an uploaded file."""
        # Mock uploaded file
        mock_file = mocker.MagicMock()
        mock_file.getbuffer.return_value = b"test content"

        # Save file
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            file_path = save_uploaded_file(mock_file, "test.pdf")

            assert os.path.exists(file_path)
            assert file_path == "temp_test.pdf"

            # Verify content
            with open(file_path, "rb") as f:
                content = f.read()
            assert content == b"test content"
        finally:
            os.chdir(original_cwd)
            if os.path.exists(os.path.join(temp_dir, "temp_test.pdf")):
                os.remove(os.path.join(temp_dir, "temp_test.pdf"))

    def test_save_uploaded_file_with_special_characters(self, mocker, temp_dir):
        """Test saving file with special characters in name."""
        mock_file = mocker.MagicMock()
        mock_file.getbuffer.return_value = b"content"

        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            file_path = save_uploaded_file(mock_file, "test file (1).pdf")
            assert "temp_test file (1).pdf" in file_path
        finally:
            os.chdir(original_cwd)


class TestCleanupTempFile:
    """Tests for cleanup_temp_file function."""

    def test_cleanup_existing_file(self, temp_dir):
        """Test cleaning up an existing file."""
        test_file = temp_dir / "test_cleanup.txt"
        test_file.write_text("test content")

        assert test_file.exists()

        cleanup_temp_file(str(test_file))

        assert not test_file.exists()

    def test_cleanup_nonexistent_file(self, temp_dir):
        """Test cleaning up a non-existent file (should not raise error)."""
        fake_path = temp_dir / "nonexistent.txt"

        # Should not raise an exception
        cleanup_temp_file(str(fake_path))

        assert not fake_path.exists()

    def test_cleanup_with_empty_path(self):
        """Test cleanup with empty path."""
        # Should handle gracefully
        cleanup_temp_file("")
