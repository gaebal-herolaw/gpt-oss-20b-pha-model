# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based Q&A system for Personal Health Agent research papers using the gpt-oss-20b model (20B parameters). The system uses GPU-accelerated inference with ChromaDB vector storage and multi-lingual (Korean + English) embeddings.

## Common Commands

### Environment Setup
```bash
# Windows activation
.\venv\Scripts\Activate.ps1

# Linux/Mac activation
source venv/bin/activate

# Install PyTorch with CUDA 12.1 support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### Development Workflow
```bash
# 1. Build vector database (required before first use)
python build_index.py

# 2. Run interactive CLI
python main.py

# 3. Run REST API server
python api_server.py

# 4. Run API with auto-reload (development)
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Individual Components
```python
# Test embedding model
python -m src.embedding_model

# Test LLM
python -m src.local_llm

# Test vector store
python -m src.vector_store

# Test RAG chain
python -m src.rag_chain

# Test data processor
python -m src.data_processor
```

### GPU Verification
```bash
# Check CUDA availability
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

### RAG Pipeline Flow
1. **Data Ingestion** (`data_processor.py`): Loads markdown files from `data/` and splits into chunks using RecursiveCharacterTextSplitter with semantic separators (H2/H3 headers, paragraphs)
2. **Embedding** (`embedding_model.py`): Uses `intfloat/multilingual-e5-large` via SentenceTransformers for GPU-accelerated batch embedding
3. **Vector Storage** (`vector_store.py`): ChromaDB persistent storage with normalized embeddings for cosine similarity search
4. **Retrieval** (`rag_chain.py`): Query → Embedding → Vector Search → Top-K contexts
5. **Generation** (`local_llm.py`): Constructs prompt with contexts → gpt-oss-20b inference → Answer

### Key Components

**LocalGPT** (`src/local_llm.py`):
- Loads gpt-oss-20b (20B params) with FP16 precision using HuggingFace Transformers
- Uses `device_map="auto"` for automatic GPU memory allocation
- Generates answers with temperature/top-p/top-k sampling
- Typically consumes ~40GB VRAM

**LocalEmbedding** (`src/embedding_model.py`):
- Wraps multilingual-e5-large model via SentenceTransformers
- Processes batches of 64 documents (configurable via `EMBEDDING_BATCH_SIZE`)
- Returns normalized embeddings for efficient cosine similarity
- Consumes ~2GB VRAM

**VectorStore** (`src/vector_store.py`):
- ChromaDB PersistentClient at `chroma_db/` directory
- Stores document chunks with metadata (source file, chunk_id)
- Batched insertion (default 2000 docs per batch)
- Query returns documents, metadatas, and distance scores

**RAGChain** (`src/rag_chain.py`):
- Orchestrates retrieval + generation pipeline
- Creates Korean-language prompts with citation instructions
- Default retrieves top-5 contexts, generates max 2048 tokens
- Returns structured response with answer, sources, and relevance scores

**DataProcessor** (`src/data_processor.py`):
- Loads all `.md` files from `data/` directory
- Splits using chunk_size=1000, chunk_overlap=200
- Preserves markdown structure by splitting on headers first
- Each chunk tagged with source filename and chunk_id

### Configuration

All settings centralized in `src/config.py`:
- **GPU Settings**: Auto-detects CUDA, falls back to CPU
- **Batch Sizes**: Embedding (64), Generation (8), Vector DB (2000)
- **Model Paths**: EMBEDDING_MODEL and LLM_MODEL (override via .env)
- **Context Length**: MAX_CONTEXT_LENGTH = 8192 tokens
- **Chunking**: CHUNK_SIZE = 1000, CHUNK_OVERLAP = 200
- **Retrieval**: TOP_K_RESULTS = 10

Edit `src/config.py` or override via `.env` file for customization.

## Data Format

Place research papers as UTF-8 encoded markdown files in `data/` directory. See `data/README.md` for Korean-language instructions. After adding new files, rebuild the index with `python build_index.py`.

## API Usage

### Endpoints
- `GET /`: Server info
- `GET /health`: Health check with vector store stats
- `POST /ask`: Submit questions (see QueryRequest model in api_server.py:38-42)

### Example Request
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "Personal Health Agent의 주요 기능은?", "k": 5, "temperature": 0.7, "max_length": 2048}'
```

## Performance Characteristics

- **Model Loading**: Embedding ~5-10s, LLM ~30-60s
- **Index Building**: 5-10 minutes for typical dataset
- **Query Latency**: Search <1s, Generation 5-15s
- **VRAM Usage**: Embedding ~2GB, LLM ~40GB, total ~42GB
- **Recommended Hardware**: NVIDIA RTX Pro 6000 96GB or equivalent

## Memory Optimization

If encountering OOM errors, reduce batch sizes in `src/config.py`:
```python
EMBEDDING_BATCH_SIZE = 32  # Default: 64
GENERATION_BATCH_SIZE = 4  # Default: 8
MAX_CONTEXT_LENGTH = 4096  # Default: 8192
```

## Entry Points

- **CLI**: `main.py` - Interactive Q&A loop
- **API**: `api_server.py` - FastAPI REST server with CORS
- **Index Builder**: `build_index.py` - Vector database construction
