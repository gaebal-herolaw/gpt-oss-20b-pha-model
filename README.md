# gpt-oss-20b-PHA Model

Personal Health Agent Q&A System using RAG (Retrieval Augmented Generation)

## 🎯 Project Overview

This project implements a RAG-based question-answering system for Personal Health Agent research papers using the gpt-oss-20b model via LM Studio.

### Key Features

- ✅ LM Studio integration with OpenAI-compatible API
- ✅ Multi-lingual embedding support (Korean + English)
- ✅ ChromaDB vector store for efficient retrieval
- ✅ OpenWebUI integration for chat interface
- ✅ FastAPI REST API server
- ✅ Interactive CLI interface
- ✅ Source tracking and citation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) installed
- gpt-oss-20b model downloaded in LM Studio
- 16GB+ RAM (32GB+ recommended)
- 20GB+ free disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# Create virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### Setup LM Studio

1. **Install LM Studio** from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download gpt-oss-20b model** (default location: `~/.lmstudio/models/`)
3. **Start LM Studio Server**:
   - Load the gpt-oss-20b model
   - Go to "Developer" or "Local Server" tab
   - Click "Start Server" (default port: 1234)

### Build Vector Database

Add your research papers (markdown files) to the `data/` folder, then:

```bash
python build_index.py
```

This will process documents and create a ChromaDB vector database in `chroma_db/`.

### Run the System

#### Option 1: OpenWebUI (Recommended)

```bash
# Start OpenWebUI
start_openwebui.bat
# or
.venv\Scripts\open-webui.exe serve
```

Then open http://localhost:8080 in your browser.

See [OPENWEBUI_SETUP.md](OPENWEBUI_SETUP.md) for detailed configuration.

#### Option 2: CLI Mode

```bash
python main.py
```

Interactive Q&A in terminal.

#### Option 3: API Server

```bash
python api_server.py
```

API will be available at http://localhost:8000

Swagger UI: http://localhost:8000/docs

## 📁 Project Structure

```
gpt-oss-20b-pha-model/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── embedding_model.py     # Embedding model (CPU)
│   ├── local_llm.py          # LM Studio API client
│   ├── data_processor.py     # Data processing
│   ├── vector_store.py       # ChromaDB vector store
│   └── rag_chain.py          # RAG pipeline
├── data/                      # Research papers (.md files)
├── chroma_db/                 # Vector database (auto-generated)
├── main.py                    # CLI interface
├── build_index.py            # Vector DB builder
├── api_server.py             # FastAPI server
├── test_search.py            # Test vector search
├── test_full_rag.py          # Test full RAG pipeline
├── openwebui_rag_function.py # OpenWebUI RAG function
├── start_openwebui.bat       # OpenWebUI launcher
├── requirements.txt          # Dependencies
├── CLAUDE.md                 # Development guide
├── OPENWEBUI_SETUP.md        # OpenWebUI setup guide
└── README.md                 # This file
```

## 🔧 Configuration

### .env Configuration

Edit `.env` to customize:

```bash
# LM Studio Server URL
LM_STUDIO_URL=http://localhost:1234/v1

# Embedding Model
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Advanced Settings

Edit `src/config.py` to customize:

- Batch sizes (default: 64 for embedding)
- Vector database parameters
- Context length (default: 8192)
- Chunk size and overlap

**Note**: Currently running in CPU mode due to Blackwell GPU (sm_120) not being supported by PyTorch 2.5.1. To enable GPU when supported, edit `src/config.py`:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📚 Usage Examples

### CLI Mode

```
질문: Personal Health Agent의 주요 기능은 무엇인가요?

[1/3] Searching for relevant documents...
[OK] Found 5 relevant chunks

[2/3] Creating prompt...
[OK] Prompt created (2341 characters)

[3/3] Generating answer...
[OK] Answer generated (856 characters)

답변:
Personal Health Agent는 다음과 같은 주요 기능을 제공합니다:
...
```

### API Mode

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of Personal Health Agent?",
    "k": 5,
    "temperature": 0.7,
    "max_length": 2048
  }'
```

### OpenWebUI Mode

1. Access http://localhost:8080
2. Add RAG Function from `openwebui_rag_function.py`
3. Ask questions in chat interface
4. System automatically searches papers and generates answers

## 🧪 Testing

### Test Vector Search Only

```bash
python test_search.py
```

### Test Full RAG Pipeline

```bash
python test_full_rag.py
```

## 🛠️ Troubleshooting

### LM Studio Not Connecting

```bash
# Check if LM Studio server is running
curl http://localhost:1234/v1/models

# Expected response: list of loaded models
```

**Solutions**:
- Ensure LM Studio is running
- Check that a model is loaded
- Verify server is started on port 1234

### Vector Database Not Found

```bash
# Rebuild the database
python build_index.py
```

### Slow Performance

**CPU Mode Optimization**:
- Reduce `EMBEDDING_BATCH_SIZE` in `src/config.py` (e.g., 32 instead of 64)
- Reduce `TOP_K_RESULTS` (e.g., 3 instead of 10)
- Use shorter `max_length` for generation

### Memory Issues

Reduce batch sizes in `src/config.py`:

```python
EMBEDDING_BATCH_SIZE = 32  # Reduce from 64
GENERATION_BATCH_SIZE = 4  # Reduce from 8
```

## 📊 Performance Benchmarks

| Task | Time | Memory Usage |
|------|------|--------------|
| Embedding Model Load | 5-10s | ~2GB RAM |
| Vector DB Build (1129 chunks) | 4-5min | ~5GB RAM |
| Query Search | <1s | ~0.5GB RAM |
| Answer Generation (LM Studio) | 5-30s | ~12GB RAM |

**Note**: Performance varies based on:
- LM Studio model quantization (MXFP4, Q4, etc.)
- Hardware specs (CPU/RAM)
- Number of documents in vector DB

## 🆕 What's New

### v2.0 (Latest)
- ✅ LM Studio integration (replaces direct model loading)
- ✅ OpenWebUI support with RAG function
- ✅ CPU mode for broader compatibility
- ✅ Improved Windows encoding support
- ✅ Test scripts for debugging
- ✅ Comprehensive documentation (CLAUDE.md, OPENWEBUI_SETUP.md)

### v1.0
- Initial release with HuggingFace Transformers
- Basic RAG pipeline
- CLI and API interfaces

## 📄 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Contact

For questions or issues, please open a GitHub issue.

## 🔗 Resources

- [LM Studio](https://lmstudio.ai/)
- [OpenWebUI](https://github.com/open-webui/open-webui)
- [ChromaDB](https://www.trychroma.com/)
- [Personal Health Agent Paper](data/personal-health-agent.md)
