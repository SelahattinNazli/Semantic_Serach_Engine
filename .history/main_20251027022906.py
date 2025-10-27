import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# Remove all HuggingFace tokens
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- Custom CSS -------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------- Session State -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ------------------- Text Splitter -------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


# ------------------- Custom Embedding Class -------------------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


# ------------------- Load Models -------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings()


@st.cache_resource
def get_vector_store(_embedding_model):
    return Chroma(
        collection_name="my_docs",
        embedding_function=_embedding_model,
        persist_directory="./chroma/db",
    )


@st.cache_resource
def load_llm():
    return ChatOllama(model="qwen3:1.7b", temperature=0.2)


# Initialize models
embedding_model = load_embedding_model()
chroma_vector_store = get_vector_store(embedding_model)
llm = load_llm()

# ------------------- Sidebar -------------------
with st.sidebar:
    st.markdown("### 🔍 Semantic Search Engine")
    st.markdown("---")

    # File Upload
    st.markdown("#### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Upload a PDF document to analyze"
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                try:
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()

                    chunks = text_splitter.split_documents(docs)
                    chroma_vector_store.add_documents(documents=chunks)

                    st.session_state.retriever = chroma_vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 5}
                    )
                    st.session_state.vector_store_ready = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []  # Clear previous chat

                    st.success("✅ Document processed!")

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    st.markdown("---")

    # Document Info
    if st.session_state.vector_store_ready:
        st.markdown("#### 📊 Current Document")
        st.info(f"**{st.session_state.pdf_name}**")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("No document loaded")

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    This tool uses:
    - **Qwen 3** for intelligent responses
    - **Sentence Transformers** for embeddings
    - **ChromaDB** for vector storage
    
    Ask questions about your uploaded PDF!
    """)

# ------------------- Main Content -------------------
# Header
st.markdown(
    '<h1 class="main-header">AI-Powered Document Q&A</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Upload a PDF and ask questions using advanced semantic search</p>',
    unsafe_allow_html=True,
)

# Main Chat Interface
if st.session_state.vector_store_ready:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs_retrieved = st.session_state.retriever.invoke(prompt)

                    system_prompt = """You are a helpful AI assistant. Answer the question: {question}

Context from the document:
{document}

Instructions:
- Use ONLY the information provided in the context above
- Be specific and detailed in your answer
- If the answer is not in the context, say "I don't know based on the provided document"
- Structure your response clearly
"""

                    prompt_template = ChatPromptTemplate.from_messages(
                        [("system", system_prompt)]
                    )

                    final_prompt = prompt_template.invoke(
                        {"question": prompt, "document": docs_retrieved}
                    )

                    response_placeholder = st.empty()
                    full_response = ""

                    for chunk in llm.stream(final_prompt):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

else:
    # Welcome screen
    st.markdown(
        """
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h2>👋 Welcome!</h2>
        <p style='font-size: 1.2rem; margin-top: 1rem;'>
            Upload a PDF document from the sidebar to start asking questions
        </p>
        <p style='margin-top: 1rem; opacity: 0.9;'>
            🚀 Powered by local AI models - Fast, Private, and Free!
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔒 Private
        All processing happens locally on your machine. Your documents never leave your computer.
        """)

    with col2:
        st.markdown("""
        ### ⚡ Fast
        Optimized for quick responses using efficient embedding models and local LLMs.
        """)

    with col3:
        st.markdown("""
        ### 🎯 Accurate
        Uses advanced semantic search to find the most relevant information in your documents.
        """)
