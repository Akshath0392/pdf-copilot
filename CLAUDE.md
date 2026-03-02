# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI-based Retrieval-Augmented Generation (RAG) pipeline that ingests PDF documents and answers questions about them using LangChain. Supports multiple LLM providers (Claude, OpenAI, Gemini) and vector store backends (FAISS, Pinecone).

## Commands

### Setup
```bash
bash init.sh          # Install all Python dependencies via pip3
cp .env.example .env  # Then edit .env with your API keys
```

### Running
```bash
python cli.py ingest              # Load PDFs from data/, chunk, embed, and store in vectorstore/
python cli.py ask "your question" # Query the ingested documents
```

### Testing
```bash
pip3 install pytest           # One-time install
python -m pytest tests/ -v    # Run all tests
```

Tests use mocks for all external dependencies (LLM APIs, HuggingFace models, FAISS, Pinecone) — no API keys or model downloads needed.

## Architecture

The pipeline flows: **PDF files → load & chunk → embed → vector store → retrieve → LLM answer**

```
cli.py          → CLI entrypoint (argparse): dispatches `ingest` and `ask` commands
loader.py       → Loads PDFs from data/ via PyPDFLoader, splits with RecursiveCharacterTextSplitter
embeddings.py   → HuggingFace embeddings factory (default: all-MiniLM-L6-v2, CPU, normalized)
store.py        → Vector store adapter: FAISS (local, saves to vectorstore/) or Pinecone (cloud)
chain.py        → LLM provider factory + RetrievalQA chain builder (stuff chain type, temperature=0)
config.py       → Centralized env config via python-dotenv with defaults
tests/          → pytest unit tests (one file per module, all dependencies mocked)
```

## Key Configuration (.env)

- `LLM_PROVIDER`: "claude" | "openai" | "gemini" (default: claude)
- `VECTOR_STORE`: "faiss" | "pinecone" (default: faiss)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Text splitting params (default: 1000 / 200)
- `EMBEDDING_MODEL`: HuggingFace model name (default: all-MiniLM-L6-v2)
- Provider API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `PINECONE_API_KEY`

## Conventions

- All LLM providers use temperature=0 for deterministic RAG responses
- Default retriever returns top k=4 chunks
- PDFs go in `data/`; FAISS index is generated in `vectorstore/` (both gitignored)
- Dependencies are managed via `init.sh`, not a requirements.txt or pyproject.toml
