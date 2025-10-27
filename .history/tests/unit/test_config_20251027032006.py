"""
Unit tests for configuration module.
"""

import pytest
from src.utils.config import (
    ModelConfig,
    ProcessingConfig,
    UIConfig,
    model_config,
    processing_config,
    ui_config,
    SYSTEM_PROMPT_TEMPLATE,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_device == "cpu"
        assert config.llm_model_name == "qwen3:1.7b"
        assert config.llm_temperature == 0.2
        assert config.collection_name == "my_docs"
        assert config.persist_directory == "./chroma/db"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            embedding_model_name="custom-model",
            llm_temperature=0.5,
            collection_name="custom_collection",
        )

        assert config.embedding_model_name == "custom-model"
        assert config.llm_temperature == 0.5
        assert config.collection_name == "custom_collection"


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default processing configuration."""
        config = ProcessingConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.add_start_index is True
        assert config.search_type == "similarity"
        assert config.retrieval_k == 5

    def test_custom_chunk_settings(self):
        """Test custom chunk settings."""
        config = ProcessingConfig(chunk_size=500, chunk_overlap=100, retrieval_k=3)

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.retrieval_k == 3


class TestUIConfig:
    """Tests for UIConfig dataclass."""

    def test_default_values(self):
        """Test default UI configuration."""
        config = UIConfig()

        assert config.page_title == "Semantic Search Engine"
        assert config.page_icon == "🔍"
        assert config.layout == "wide"
        assert config.sidebar_state == "expanded"

    def test_custom_ui_settings(self):
        """Test custom UI settings."""
        config = UIConfig(page_title="Custom Title", layout="centered")

        assert config.page_title == "Custom Title"
        assert config.layout == "centered"


class TestGlobalConfigs:
    """Tests for global configuration instances."""

    def test_model_config_instance(self):
        """Test global model_config instance."""
        assert isinstance(model_config, ModelConfig)
        assert model_config.llm_model_name == "qwen3:1.7b"

    def test_processing_config_instance(self):
        """Test global processing_config instance."""
        assert isinstance(processing_config, ProcessingConfig)
        assert processing_config.chunk_size == 1000

    def test_ui_config_instance(self):
        """Test global ui_config instance."""
        assert isinstance(ui_config, UIConfig)
        assert ui_config.page_icon == "🔍"


class TestSystemPromptTemplate:
    """Tests for system prompt template."""

    def test_prompt_template_exists(self):
        """Test that system prompt template is defined."""
        assert SYSTEM_PROMPT_TEMPLATE is not None
        assert isinstance(SYSTEM_PROMPT_TEMPLATE, str)
        assert len(SYSTEM_PROMPT_TEMPLATE) > 0

    def test_prompt_contains_placeholders(self):
        """Test that prompt template contains required placeholders."""
        assert "{question}" in SYSTEM_PROMPT_TEMPLATE
        assert "{document}" in SYSTEM_PROMPT_TEMPLATE

    def test_prompt_contains_instructions(self):
        """Test that prompt template contains instructions."""
        assert "ONLY" in SYSTEM_PROMPT_TEMPLATE
        assert "context" in SYSTEM_PROMPT_TEMPLATE.lower()
