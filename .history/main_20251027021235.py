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

# ------------------- Session State Başlatma -------------------
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------- Text Splitter -------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


# ------------------- Embedding Model (Free & Local) -------------------
@st.cache_resource
def load_embedding_model():
    # Token sorununu çözmek için use_auth_token=False ekledik
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embedding_model = load_embedding_model()


# ------------------- Vector Store -------------------
@st.cache_resource
def get_vector_store(_embedding_model):
    return Chroma(
        collection_name="my_docs",
        embedding_function=_embedding_model,
        persist_directory="./chroma/db",
    )


chroma_vector_store = get_vector_store(embedding_model)


# ------------------- LLM Model (Ollama - Local) -------------------
@st.cache_resource
def load_llm():
    return ChatOllama(model="qwen:1.7b", temperature=0.2)


llm = load_llm()

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Select a PDF file:")

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            # Temporarily save the PDF
            temp_file_path = f"temp_{uploaded_file.name}"
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

            # Create retriever ve session state'e kaydet
            st.session_state.retriever = chroma_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            st.session_state.vector_store_ready = True

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback

            st.error(traceback.format_exc())

        finally:
            # Delete temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# ------------------- Chat Interface -------------------
if st.session_state.vector_store_ready:
    st.success("✅ PDF işlendi! Şimdi soru sorabilirsiniz.")

    if prompt := st.chat_input("Ask a question (e.g., What is this document about?)"):
        with st.spinner("Thinking..."):
            try:
                docs_retrieved = st.session_state.retriever.invoke(prompt)

                system_prompt = """You're a helpful assistant. Please answer the following question: {question}
                using ONLY the information from: {document}.
                If you cannot find the answer, say "I don't know based on the provided document."
                """

                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", system_prompt)]
                )

                final_prompt = prompt_template.invoke(
                    {"question": prompt, "document": docs_retrieved}
                )

                st.subheader("💬 Model Response:")
                result_placeholder = st.empty()
                full_completion = ""

                for chunk in llm.stream(final_prompt):
                    full_completion += chunk.content
                    result_placeholder.write(full_completion)

            except Exception as e:
                st.error(f"Error during query: {e}")
                import traceback

                st.error(traceback.format_exc())
else:
    st.info("👆 Lütfen önce bir PDF dosyası yükleyin.")
