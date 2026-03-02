import os
import importlib
import unittest.mock as mock

_ALL_CONFIG_KEYS = [
    "VECTOR_STORE", "LLM_PROVIDER", "LLM_MODEL",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
    "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
    "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
]


def _reload_config(**env_overrides):
    """Reload config module with specific env vars set."""
    with mock.patch.dict(os.environ, clear=False) as patched_env:
        # Remove all config-relevant keys so os.getenv falls through to defaults
        for key in _ALL_CONFIG_KEYS:
            patched_env.pop(key, None)
        # Apply overrides
        patched_env.update(env_overrides)
        # Prevent load_dotenv from loading a real .env file
        with mock.patch("dotenv.load_dotenv"):
            import config
            importlib.reload(config)
            return config


def test_defaults():
    cfg = _reload_config()
    assert cfg.VECTOR_STORE == "faiss"
    assert cfg.LLM_PROVIDER == "claude"
    assert cfg.CHUNK_SIZE == 1000
    assert cfg.CHUNK_OVERLAP == 200
    assert cfg.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert cfg.PINECONE_INDEX_NAME == "rag-index"


def test_custom_env_overrides():
    cfg = _reload_config(
        VECTOR_STORE="pinecone",
        LLM_PROVIDER="openai",
        CHUNK_SIZE="500",
        CHUNK_OVERLAP="50",
        EMBEDDING_MODEL="custom-model",
    )
    assert cfg.VECTOR_STORE == "pinecone"
    assert cfg.LLM_PROVIDER == "openai"
    assert cfg.CHUNK_SIZE == 500
    assert cfg.CHUNK_OVERLAP == 50
    assert cfg.EMBEDDING_MODEL == "custom-model"


def test_default_models_per_provider():
    cfg = _reload_config(LLM_PROVIDER="claude")
    assert cfg.LLM_MODEL == "claude-sonnet-4-5-20250929"

    cfg = _reload_config(LLM_PROVIDER="openai")
    assert cfg.LLM_MODEL == "gpt-4o"

    cfg = _reload_config(LLM_PROVIDER="gemini")
    assert cfg.LLM_MODEL == "gemini-2.0-flash"


def test_explicit_model_overrides_default():
    cfg = _reload_config(LLM_PROVIDER="claude", LLM_MODEL="my-custom-model")
    assert cfg.LLM_MODEL == "my-custom-model"


def test_data_and_vectorstore_dirs():
    cfg = _reload_config()
    project_root = os.path.dirname(os.path.dirname(__file__))
    assert cfg.DATA_DIR == os.path.join(project_root, "data")
    assert cfg.VECTORSTORE_DIR == os.path.join(project_root, "vectorstore")
