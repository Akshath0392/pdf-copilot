import sys
from unittest import mock

import pytest


@mock.patch("store.get_embeddings")
@mock.patch("store.FAISS")
def test_create_vectorstore_faiss(mock_faiss_cls, mock_emb):
    mock_emb.return_value = mock.MagicMock()
    mock_vs = mock.MagicMock()
    mock_faiss_cls.from_documents.return_value = mock_vs

    with mock.patch("store.VECTOR_STORE", "faiss"):
        from store import create_vectorstore
        result = create_vectorstore([mock.MagicMock()])

    mock_faiss_cls.from_documents.assert_called_once()
    mock_vs.save_local.assert_called_once()
    assert result is mock_vs


@mock.patch("store.get_embeddings")
@mock.patch("store.VECTOR_STORE", "pinecone")
@mock.patch("store.PINECONE_API_KEY", "fake-pinecone-key")
@mock.patch("store.PINECONE_INDEX_NAME", "test-index")
def test_create_vectorstore_pinecone(mock_emb):
    mock_emb.return_value = mock.MagicMock()
    mock_pinecone_cls = mock.MagicMock()
    mock_pinecone_vs = mock.MagicMock()
    mock_pinecone_cls.from_documents.return_value = mock_pinecone_vs

    mock_module = mock.MagicMock()
    mock_module.PineconeVectorStore = mock_pinecone_cls

    with mock.patch.dict(sys.modules, {"langchain_pinecone": mock_module}):
        from store import create_vectorstore
        result = create_vectorstore([mock.MagicMock()])

    mock_pinecone_cls.from_documents.assert_called_once()
    assert result is mock_pinecone_vs


@mock.patch("store.get_embeddings")
@mock.patch("store.FAISS")
def test_load_vectorstore_faiss_success(mock_faiss_cls, mock_emb):
    mock_emb.return_value = mock.MagicMock()
    mock_vs = mock.MagicMock()
    mock_faiss_cls.load_local.return_value = mock_vs

    with mock.patch("store.VECTOR_STORE", "faiss"), \
         mock.patch("os.path.exists", return_value=True):
        from store import load_vectorstore
        result = load_vectorstore()

    mock_faiss_cls.load_local.assert_called_once()
    assert result is mock_vs


@mock.patch("store.get_embeddings")
def test_load_vectorstore_faiss_missing_index(mock_emb):
    mock_emb.return_value = mock.MagicMock()

    with mock.patch("store.VECTOR_STORE", "faiss"), \
         mock.patch("os.path.exists", return_value=False):
        from store import load_vectorstore

        with pytest.raises(FileNotFoundError, match="No FAISS index found"):
            load_vectorstore()


@mock.patch("store.get_embeddings")
@mock.patch("store.VECTOR_STORE", "pinecone")
@mock.patch("store.PINECONE_API_KEY", "fake-pinecone-key")
@mock.patch("store.PINECONE_INDEX_NAME", "test-index")
def test_load_vectorstore_pinecone(mock_emb):
    mock_emb.return_value = mock.MagicMock()
    mock_pinecone_cls = mock.MagicMock()
    mock_pinecone_vs = mock.MagicMock()
    mock_pinecone_cls.from_existing_index.return_value = mock_pinecone_vs

    mock_module = mock.MagicMock()
    mock_module.PineconeVectorStore = mock_pinecone_cls

    with mock.patch.dict(sys.modules, {"langchain_pinecone": mock_module}):
        from store import load_vectorstore
        result = load_vectorstore()

    mock_pinecone_cls.from_existing_index.assert_called_once()
    assert result is mock_pinecone_vs


@mock.patch("store.load_vectorstore")
def test_get_retriever_default_k(mock_load):
    mock_vs = mock.MagicMock()
    mock_load.return_value = mock_vs
    mock_retriever = mock.MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    from store import get_retriever
    result = get_retriever()

    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
    assert result is mock_retriever


@mock.patch("store.load_vectorstore")
def test_get_retriever_custom_k(mock_load):
    mock_vs = mock.MagicMock()
    mock_load.return_value = mock_vs

    from store import get_retriever
    get_retriever(k=10)

    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
