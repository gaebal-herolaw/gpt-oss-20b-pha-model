# gpt-oss-20b-PHA Model

Personal Health Agent Q&A System using RAG (Retrieval Augmented Generation)

## 🎯 Project Overview

This project implements a RAG-based question-answering system for Personal Health Agent research papers using the gpt-oss-20b model.

### Key Features

- ✅ Local GPU-accelerated inference
- ✅ Multi-lingual embedding support (Korean + English)
- ✅ ChromaDB vector store for efficient retrieval
- ✅ FastAPI REST API server
- ✅ Interactive CLI interface
- ✅ Source tracking and citation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: RTX Pro 6000 96GB)
- CUDA 11.8+ or 12.1+
- 32GB+ RAM
- 100GB+ free disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/gaebal-herolaw/gpt-oss-20b-pha-model.git
cd gpt-oss-20b-pha-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### Build Vector Database

```bash
python build_index.py
```

### Run CLI

```bash
python main.py
```

### Run API Server

```bash
python api_server.py
```

API will be available at `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

## 📁 Project Structure

```
gpt-oss-20b-pha-model/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── embedding_model.py     # Embedding model
│   ├── local_llm.py          # Local LLM
│   ├── data_processor.py     # Data processing
│   ├── vector_store.py       # Vector store
│   └── rag_chain.py          # RAG pipeline
├── data/                      # Research papers (add your files here)
├── main.py                    # CLI interface
├── build_index.py            # Vector DB builder
├── api_server.py             # FastAPI server
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

- Model paths and settings
- Batch sizes for GPU optimization
- Vector database parameters
- API server settings

## 📚 Usage Examples

### CLI Mode

```python
질문: Personal Health Agent의 주요 기능은 무엇인가요?

답변:
Personal Health Agent는 다음과 같은 주요 기능을 제공합니다:
...
```

### API Mode

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key features of Personal Health Agent?", "k": 5}'
```

## 🛠️ Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

Reduce batch sizes in `src/config.py`:

```python
EMBEDDING_BATCH_SIZE = 32  # Reduce from 64
GENERATION_BATCH_SIZE = 4  # Reduce from 8
```

## 📊 Performance Benchmarks

| Task | Time | VRAM Usage |
|------|------|-----------|
| Embedding Model Load | 5-10s | ~2GB |
| LLM Load (20B) | 30-60s | ~40GB |
| Vector DB Build | 5-10min | ~5GB |
| Query Search | <1s | ~0.1GB |
| Answer Generation | 5-15s | ~42GB |

## 📄 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Contact

For questions or issues, please open a GitHub issue.
