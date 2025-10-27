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

# ------------------- Streamlit UI -------------------
st.title("Local Semantic Search Engine")
st.header(
    "Upload a PDF and perform semantic search with the Qwen model!", divider="green"
)

# ------------------- Session State -------------------
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------- Text Splitter -------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


# ------------------- Custom Embedding Class -------------------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        """Embed a single query."""
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


# Initialize
embedding_model = load_embedding_model()
chroma_vector_store = get_vector_store(embedding_model)
llm = load_llm()

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Select a PDF file:")

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

            chunks = text_splitter.split_documents(docs)
            st.success(f"{len(chunks)} text chunks created successfully ✅")

            chroma_vector_store.add_documents(documents=chunks)
            st.info("Vector database created successfully 🔍")

            st.session_state.retriever = chroma_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            st.session_state.vector_store_ready = True

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback

            st.error(traceback.format_exc())

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# ------------------- Chat Interface -------------------
if st.session_state.vector_store_ready:
    st.success("✅ PDF processed successfully! You can now ask questions.")

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
    st.info("👆 Please upload a PDF file first.")
