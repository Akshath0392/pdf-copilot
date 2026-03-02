from unittest import mock

import pytest


def test_main_ingest_calls_ingest():
    with mock.patch("sys.argv", ["cli.py", "ingest"]), \
         mock.patch("cli.ingest") as mock_ingest:
        from cli import main
        main()
        mock_ingest.assert_called_once()


def test_main_ask_calls_ask():
    with mock.patch("sys.argv", ["cli.py", "ask", "What is AI?"]), \
         mock.patch("cli.ask") as mock_ask:
        from cli import main
        main()
        mock_ask.assert_called_once_with("What is AI?")


def test_main_no_command_exits_with_1():
    with mock.patch("sys.argv", ["cli.py"]):
        from cli import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_ask_missing_question_exits():
    with mock.patch("sys.argv", ["cli.py", "ask"]):
        from cli import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2  # argparse exits with 2 for missing args


def test_ingest_function():
    mock_load = mock.MagicMock(return_value=["chunk1", "chunk2"])
    mock_create = mock.MagicMock()

    with mock.patch("loader.load_and_split", mock_load), \
         mock.patch("store.create_vectorstore", mock_create):
        from cli import ingest
        ingest()

    mock_load.assert_called_once()
    mock_create.assert_called_once_with(["chunk1", "chunk2"])


def test_ask_function_prints_answer(capsys):
    mock_chain = mock.MagicMock()
    mock_chain.invoke.return_value = {
        "result": "42 is the answer",
        "source_documents": [
            mock.MagicMock(metadata={"source": "doc.pdf", "page": 1}),
        ],
    }

    with mock.patch("chain.get_qa_chain", return_value=mock_chain):
        from cli import ask
        ask("What is the answer?")

    captured = capsys.readouterr()
    assert "42 is the answer" in captured.out
    assert "doc.pdf" in captured.out
    assert "page 1" in captured.out
