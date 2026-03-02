from unittest import mock


@mock.patch("embeddings.HuggingFaceEmbeddings")
def test_get_embeddings_returns_instance(mock_hf):
    mock_hf.return_value = mock.MagicMock()

    from embeddings import get_embeddings
    result = get_embeddings()

    mock_hf.assert_called_once_with(
        model_name=mock.ANY,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    assert result is mock_hf.return_value


@mock.patch("embeddings.HuggingFaceEmbeddings")
def test_get_embeddings_uses_configured_model(mock_hf):
    with mock.patch("embeddings.EMBEDDING_MODEL", "custom-embed-model"):
        from embeddings import get_embeddings
        get_embeddings()

    call_kwargs = mock_hf.call_args
    assert call_kwargs[1]["model_name"] == "custom-embed-model" or \
           call_kwargs[0][0] == "custom-embed-model" if call_kwargs[0] else \
           call_kwargs.kwargs.get("model_name") == "custom-embed-model"
