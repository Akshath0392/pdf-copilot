# RAG Pipeline

A command-line Retrieval-Augmented Generation (RAG) pipeline that lets you ingest PDF documents and ask questions against them using LangChain.

## Features

- **PDF ingestion** — loads all PDFs from the `data/` directory, splits them into chunks, and stores embeddings in a vector database
- **Multi-provider LLM support** — choose between Claude, OpenAI, or Gemini as the answering model
- **Vector store options** — FAISS (local, default) or Pinecone (cloud)
- **HuggingFace embeddings** — uses `all-MiniLM-L6-v2` by default (runs on CPU)

## Project Structure

```
.
├── cli.py           # CLI entrypoint (ingest / ask commands)
├── chain.py         # LLM selection and RetrievalQA chain
├── config.py        # Environment variable loading and defaults
├── embeddings.py    # HuggingFace embedding wrapper
├── loader.py        # PDF loading and text chunking
├── store.py         # Vector store creation, loading, and retrieval
├── init.sh          # Dependency installation script
├── data/            # Place your PDF files here
├── vectorstore/     # FAISS index output (auto-generated)
├── .env.example     # Sample environment configuration
└── .env             # Your local configuration (not committed)
```

## Setup

### 1. Install dependencies

```bash
bash init.sh
```

### 2. Configure environment variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `claude`, `openai`, or `gemini` | `claude` |
| `LLM_MODEL` | Model name for the chosen provider | Provider-specific default |
| `ANTHROPIC_API_KEY` | API key (required if provider is `claude`) | — |
| `OPENAI_API_KEY` | API key (required if provider is `openai`) | — |
| `GOOGLE_API_KEY` | API key (required if provider is `gemini`) | — |
| `VECTOR_STORE` | `faiss` or `pinecone` | `faiss` |
| `PINECONE_API_KEY` | Required only if using Pinecone | — |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-index` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Characters per text chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

## Usage

### Ingest documents

Place your PDF files in the `data/` directory, then run:

```bash
python cli.py ingest
```

This loads the PDFs, splits them into chunks, generates embeddings, and saves the vector index.

### Ask a question

```bash
python cli.py ask "What are the key findings in the report?"
```

The pipeline retrieves the most relevant chunks and uses the configured LLM to generate an answer, along with source references.
