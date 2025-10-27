# 🔍 Semantic Search Engine

A professional, production-ready semantic search engine for PDF documents using **local AI models**.

<img width="1500" height="799" alt="Screenshot 2025-10-27 at 2 53 43 AM" src="https://github.com/user-attachments/assets/a07a3d39-f93c-4b7e-9352-df0d936b328c" />

<img width="1471" height="800" alt="Screenshot 2025-10-27 at 2 55 21 AM" src="https://github.com/user-attachments/assets/4d08ea83-ea52-4182-86b6-2ecdbd9a46cf" />

<img width="1470" height="601" alt="Screenshot 2025-10-27 at 1 38 23 PM" src="https://github.com/user-attachments/assets/ce85ca29-3e70-4692-8fe8-3d57993ad6a6" />


---

## ✨ Features

- 🔒 **100% Local & Private** — All processing happens on your machine  
- ⚡ **Fast & Efficient** — Optimized for quick responses  
- 🎯 **Accurate Semantic Search** — Advanced embedding models  
- 💬 **Chat Interface** — Natural conversation with your documents  
- 🐳 **Docker Ready** — Easy deployment anywhere  
- 🧪 **Well Tested** — 50+ unit and integration tests  

---

## 🚀 Quick Start

### **Option 1: Docker (Recommended)**

```bash
# Using Docker Compose
docker-compose up

# Or using Docker directly
docker build -t semantic-search .
docker run -p 8501:8501 semantic-search


Visit: http://localhost:8501
```

### Option 2: Local Development
**Install dependencies**
```bash
uv sync
```
**Run the application**
```bash
uv run streamlit run main.py
```

### Prerequisites

**For Docker:** Docker & Docker Compose

**For Local Development:** Python 3.12+

**UV package manager**

**Ollama with qwen3:1.7b model**

**Install Ollama Model:** ollama pull qwen3:1.7b


### Project Structure
```bash
Semantic_Search_Engine_Free/
├── src/
│   ├── models/          # AI models (embeddings, LLM, vector store)
│   ├── utils/           # Utilities (config, PDF processor)
│   └── ui/              # UI components (chat, sidebar, styles)
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── main.py              # Application entry point
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── pyproject.toml       # Project dependencies
```

### Running Tests
**Run all tests**
```bash
uv run pytest
```
**Run with coverage**
```bash
uv run pytest --cov=src --cov-report=html
```

**View coverage report**

```bash
open htmlcov/index.html
```

## Configuration

You can customize the application by editing:
```bash
src/utils/config.py
```

## Adjust the following as needed:

Embedding Model — Change the Sentence Transformer model

LLM Model — Switch between different Ollama models

Chunk Size — Adjust document chunking parameters

Retrieval Settings — Modify semantic search options

**Note:**
This project is intentionally designed to use completely free and local models for both the LLM chat model and embedding model.
However, you can easily integrate your own preferred LLM or embedding model (e.g., OpenAI, Anthropic, or Hugging Face models) by updating config.py.

## 🐳 Docker Details
```bash
Build
docker build -t semantic-search-engine .

Run
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/chroma:/app/chroma \
  --name semantic-search \
  semantic-search-engine

Environment Variables
Variable	Description	Default
STREAMLIT_SERVER_PORT	Port number	8501
STREAMLIT_SERVER_ADDRESS	Server address	0.0.0.0
```

## 🤝 Contributing

Fork the repository

Create your feature branch
```bash
git checkout -b feature/AmazingFeature
```

Commit your changes
```bash
git commit -m 'Add some AmazingFeature'
```

Push to the branch
```bash
git push origin feature/AmazingFeature
```

Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Qwen
 — LLM model

Sentence Transformers
 — Embedding models

ChromaDB
 — Vector database

Streamlit
 — Web framework

LangChain
 — AI framework
