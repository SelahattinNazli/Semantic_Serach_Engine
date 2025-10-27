import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ------------------- Streamlit UI -------------------
st.title("Local Semantic Search Engine")
st.header(
    "Upload a PDF and perform semantic search with the Qwen model!", divider="green"
)

# ------------------- Text Splitter -------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# ------------------- Embedding Model (Free & Local) -------------------
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# ------------------- Vector Store -------------------
chroma_vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma/db",
)

# ------------------- LLM Model (Ollama - Local) -------------------
llm = ChatOllama(model="qwen:1.7b", temperature=0.2)
# For faster testing:
# llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.2)

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Select a PDF file:")

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            # Temporarily save the PDF
            temp_file_path = uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            # Split PDF into chunks
            chunks = text_splitter.split_documents(docs)
            st.success(f"{len(chunks)} text chunks created ✅")

            # Add to Chroma (embedding + indexing)
            chroma_vector_store.add_documents(documents=chunks)
            st.info("Vector database successfully created 🔍")

            # Create retriever
            retriever = chroma_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )

            # Get query from user
            if prompt := st.chat_input(
                "Ask a question (e.g., What is this document about?)"
            ):
                docs_retrieved = retriever.invoke(prompt)

                # Prompt Template
                system_prompt = """You're a helpful assistant. Please answer the following question: {question}
                using ONLY the information from: {document}.
                If you cannot find the answer, say "I don’t know based on the provided document."
                """

                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", system_prompt)]
                )

                final_prompt = prompt_template.invoke(
                    {"question": prompt, "document": docs_retrieved}
                )

                # Display the result as a stream
                st.subheader("💬 Model Response:")
                result_placeholder = st.empty()
                full_completion = ""

                for chunk in llm.stream(final_prompt):
                    full_completion += chunk.content
                    result_placeholder.write(full_completion)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Delete temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
