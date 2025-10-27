"""
Configuration settings for the Semantic Search Engine.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    # Embedding Model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # LLM Model
    llm_model_name: str = "qwen3:1.7b"
    llm_temperature: float = 0.2

    # Vector Store
    collection_name: str = "my_docs"
    persist_directory: str = "./chroma/db"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    add_start_index: bool = True

    # Retrieval
    search_type: Literal["similarity", "mmr"] = "similarity"
    retrieval_k: int = 5


@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""

    page_title: str = "Semantic Search Engine"
    page_icon: str = "🔍"
    layout: Literal["centered", "wide"] = "wide"
    sidebar_state: Literal["expanded", "collapsed"] = "expanded"


# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the question: {question}

Context from the document:
{document}

Instructions:
- Use ONLY the information provided in the context above
- Be specific and detailed in your answer
- If the answer is not in the context, say "I don't know based on the provided document"
- Structure your response clearly with bullet points when appropriate
"""


# Create default configurations
model_config = ModelConfig()
processing_config = ProcessingConfig()
ui_config = UIConfig()
