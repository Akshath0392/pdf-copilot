import sys
from unittest import mock

import pytest


@mock.patch("chain.LLM_PROVIDER", "claude")
@mock.patch("chain.LLM_MODEL", "test-claude-model")
@mock.patch("chain.ANTHROPIC_API_KEY", "fake-key")
def test_get_llm_claude():
    mock_module = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"langchain_anthropic": mock_module}):
        from chain import get_llm
        get_llm()

    mock_module.ChatAnthropic.assert_called_once_with(
        model="test-claude-model",
        anthropic_api_key="fake-key",
        temperature=0,
    )


@mock.patch("chain.LLM_PROVIDER", "openai")
@mock.patch("chain.LLM_MODEL", "test-openai-model")
@mock.patch("chain.OPENAI_API_KEY", "fake-key")
def test_get_llm_openai():
    mock_module = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"langchain_openai": mock_module}):
        from chain import get_llm
        get_llm()

    mock_module.ChatOpenAI.assert_called_once_with(
        model="test-openai-model",
        openai_api_key="fake-key",
        temperature=0,
    )


@mock.patch("chain.LLM_PROVIDER", "gemini")
@mock.patch("chain.LLM_MODEL", "test-gemini-model")
@mock.patch("chain.GOOGLE_API_KEY", "fake-key")
def test_get_llm_gemini():
    mock_module = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"langchain_google_genai": mock_module}):
        from chain import get_llm
        get_llm()

    mock_module.ChatGoogleGenerativeAI.assert_called_once_with(
        model="test-gemini-model",
        google_api_key="fake-key",
        temperature=0,
    )


def test_get_llm_unknown_provider():
    with mock.patch("chain.LLM_PROVIDER", "unknown"):
        from chain import get_llm
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm()


@mock.patch("chain.get_retriever")
@mock.patch("chain.get_llm")
def test_get_qa_chain_returns_chain(mock_get_llm, mock_get_retriever):
    mock_get_llm.return_value = mock.MagicMock()
    mock_get_retriever.return_value = mock.MagicMock()

    with mock.patch("chain.RetrievalQA") as mock_qa:
        mock_qa.from_chain_type.return_value = mock.MagicMock()

        from chain import get_qa_chain, PROMPT
        result = get_qa_chain()

        mock_qa.from_chain_type.assert_called_once_with(
            llm=mock_get_llm.return_value,
            chain_type="stuff",
            retriever=mock_get_retriever.return_value,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
