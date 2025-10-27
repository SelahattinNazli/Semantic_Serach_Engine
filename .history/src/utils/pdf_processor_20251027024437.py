"""
PDF document processing utilities.
"""

import os
from typing import List, BinaryIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """Handle PDF loading and text chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
    ):
        """
        Initialize the PDF processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            add_start_index: Whether to add start index to chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents
        """
        return self.text_splitter.split_documents(documents)

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Load and split a PDF in one step.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of chunked documents
        """
        documents = self.load_pdf(file_path)
        return self.split_documents(documents)


def save_uploaded_file(uploaded_file: BinaryIO, filename: str) -> str:
    """
    Save an uploaded file temporarily.

    Args:
        uploaded_file: Streamlit uploaded file object
        filename: Name for the temporary file

    Returns:
        Path to the saved file
    """
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def cleanup_temp_file(file_path: str) -> None:
    """
    Remove a temporary file.

    Args:
        file_path: Path to the file to remove
    """
    if os.path.exists(file_path):
        os.remove(file_path)
