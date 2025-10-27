"""
Pytest configuration and shared fixtures.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences and paragraphs.
    
    The document discusses artificial intelligence and machine learning.
    These technologies are transforming many industries.
    
    Testing is an important part of software development.
    Unit tests help ensure code quality and reliability.
    """


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for testing."""
    return [
        Document(
            page_content="This is the first test document about AI.",
            metadata={"source": "test1.pdf", "page": 0},
        ),
        Document(
            page_content="This is the second test document about machine learning.",
            metadata={"source": "test2.pdf", "page": 1},
        ),
        Document(
            page_content="This document discusses natural language processing.",
            metadata={"source": "test3.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    # Create a simple text file (in real tests, you'd use a proper PDF)
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_text("Sample PDF content for testing")
    return pdf_path


@pytest.fixture
def mock_embedding_model(mocker):
    """Mock embedding model for testing."""
    mock = mocker.MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_llm(mocker):
    """Mock LLM for testing."""
    mock = mocker.MagicMock()
    mock.stream.return_value = iter(["Response ", "from ", "LLM"])
    return mock


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables before each test."""
    env_vars = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
    for var in env_vars:
        os.environ.pop(var, None)
    yield
