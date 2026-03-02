import json
from unittest import mock

import pytest
from pydantic import ValidationError

from agent import AgentResponse, SourceInfo, parse_response


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------

class TestSourceInfo:
    def test_valid(self):
        s = SourceInfo(document="report.pdf", page=3)
        assert s.document == "report.pdf"
        assert s.page == 3

    def test_json_roundtrip(self):
        s = SourceInfo(document="file.pdf", page=1)
        data = json.loads(s.model_dump_json())
        assert SourceInfo(**data) == s


class TestAgentResponse:
    def test_valid_full(self):
        r = AgentResponse(
            answer="The answer is 42.",
            confidence="high",
            sources=[SourceInfo(document="guide.pdf", page=5)],
            follow_up_questions=["What about section 3?"],
        )
        assert r.answer == "The answer is 42."
        assert r.confidence == "high"
        assert len(r.sources) == 1
        assert len(r.follow_up_questions) == 1

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            AgentResponse(answer="x", confidence="very_high")

    def test_defaults(self):
        r = AgentResponse(answer="text", confidence="low")
        assert r.sources == []
        assert r.follow_up_questions == []

    def test_json_roundtrip(self):
        r = AgentResponse(
            answer="ans",
            confidence="medium",
            sources=[SourceInfo(document="a.pdf", page=1)],
            follow_up_questions=["q1"],
        )
        data = json.loads(r.model_dump_json())
        assert AgentResponse(**data) == r


# ---------------------------------------------------------------------------
# parse_response tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def _make_json(self, **overrides):
        data = {
            "answer": "test answer",
            "confidence": "high",
            "sources": [{"document": "doc.pdf", "page": 1}],
            "follow_up_questions": ["follow up?"],
        }
        data.update(overrides)
        return json.dumps(data)

    def test_clean_json(self):
        raw = self._make_json()
        resp = parse_response(raw)
        assert resp.answer == "test answer"
        assert resp.confidence == "high"

    def test_json_with_whitespace(self):
        raw = "  \n" + self._make_json() + "\n  "
        resp = parse_response(raw)
        assert resp.answer == "test answer"

    def test_markdown_fences(self):
        raw = "Here is the result:\n```json\n" + self._make_json() + "\n```"
        resp = parse_response(raw)
        assert resp.answer == "test answer"

    def test_markdown_fences_no_lang(self):
        raw = "Result:\n```\n" + self._make_json() + "\n```"
        resp = parse_response(raw)
        assert resp.answer == "test answer"

    def test_embedded_json_in_text(self):
        raw = "I found this: " + self._make_json() + " Hope that helps!"
        resp = parse_response(raw)
        assert resp.answer == "test answer"

    def test_fallback_plain_text(self):
        raw = "I don't know the answer to that question."
        resp = parse_response(raw)
        assert resp.answer == raw
        assert resp.confidence == "low"
        assert resp.sources == []
        assert resp.follow_up_questions == []

    def test_invalid_json_falls_back(self):
        raw = '{"answer": "test", "confidence": "invalid_value"}'
        resp = parse_response(raw)
        # The embedded JSON has an invalid confidence, so it should fall back
        assert resp.confidence == "low"
        assert resp.answer == raw


# ---------------------------------------------------------------------------
# search_compliance_docs tests
# ---------------------------------------------------------------------------

class TestSearchComplianceDocs:
    @mock.patch("agent.get_retriever")
    def test_returns_formatted_output(self, mock_get_retriever):
        mock_doc = mock.MagicMock()
        mock_doc.metadata = {"source": "rbi_circular.pdf", "page": 5}
        mock_doc.page_content = "Banks must comply with KYC norms."

        mock_retriever = mock.MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        from agent import search_compliance_docs
        result = search_compliance_docs.invoke("KYC requirements")

        assert "rbi_circular.pdf" in result
        assert "Page: 5" in result
        assert "KYC norms" in result

    @mock.patch("agent.get_retriever")
    def test_empty_results(self, mock_get_retriever):
        mock_retriever = mock.MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from agent import search_compliance_docs
        result = search_compliance_docs.invoke("nonexistent topic")

        assert result == "No relevant documents found."

    @mock.patch("agent.get_retriever")
    def test_missing_metadata(self, mock_get_retriever):
        mock_doc = mock.MagicMock()
        mock_doc.metadata = {}
        mock_doc.page_content = "Some content"

        mock_retriever = mock.MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        from agent import search_compliance_docs
        result = search_compliance_docs.invoke("query")

        assert "unknown" in result
        assert "?" in result


# ---------------------------------------------------------------------------
# build_agent tests
# ---------------------------------------------------------------------------

class TestBuildAgent:
    @mock.patch("agent.AgentExecutor")
    @mock.patch("agent.create_tool_calling_agent")
    @mock.patch("agent.get_llm")
    def test_returns_agent_executor(self, mock_get_llm, mock_create_agent, mock_ae_cls):
        mock_get_llm.return_value = mock.MagicMock()
        mock_create_agent.return_value = mock.MagicMock()
        mock_ae_cls.return_value = mock.MagicMock()

        from agent import build_agent
        executor = build_agent()

        mock_ae_cls.assert_called_once()
        assert executor is mock_ae_cls.return_value

    @mock.patch("agent.AgentExecutor")
    @mock.patch("agent.create_tool_calling_agent")
    @mock.patch("agent.get_llm")
    def test_create_agent_called_with_prompt(self, mock_get_llm, mock_create_agent, mock_ae_cls):
        mock_get_llm.return_value = mock.MagicMock()
        mock_create_agent.return_value = mock.MagicMock()
        mock_ae_cls.return_value = mock.MagicMock()

        from agent import build_agent
        build_agent()

        mock_create_agent.assert_called_once()
        args = mock_create_agent.call_args[0]
        prompt = args[2]
        assert "input" in prompt.input_variables


# ---------------------------------------------------------------------------
# chat_repl tests
# ---------------------------------------------------------------------------

class TestChatRepl:
    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["exit"])
    def test_exit_command(self, mock_input, mock_build, capsys):
        from agent import chat_repl
        chat_repl()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["quit"])
    def test_quit_command(self, mock_input, mock_build, capsys):
        from agent import chat_repl
        chat_repl()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=EOFError)
    def test_eof_exits(self, mock_input, mock_build, capsys):
        from agent import chat_repl
        chat_repl()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_ctrl_c_exits(self, mock_input, mock_build, capsys):
        from agent import chat_repl
        chat_repl()
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["What is KYC?", "exit"])
    def test_agent_invocation(self, mock_input, mock_build, capsys):
        captured_args = []

        def capture_invoke(args):
            captured_args.append({
                "input": args["input"],
                "history_len": len(args["chat_history"]),
            })
            return {
                "output": json.dumps({
                    "answer": "KYC is Know Your Customer",
                    "confidence": "high",
                    "sources": [{"document": "rbi.pdf", "page": 2}],
                    "follow_up_questions": ["What are KYC documents?"],
                })
            }

        mock_executor = mock.MagicMock()
        mock_executor.invoke.side_effect = capture_invoke
        mock_build.return_value = mock_executor

        from agent import chat_repl
        chat_repl()

        assert len(captured_args) == 1
        assert captured_args[0]["input"] == "What is KYC?"
        assert captured_args[0]["history_len"] == 0

        captured = capsys.readouterr()
        assert "KYC is Know Your Customer" in captured.out

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["q1", "q2", "exit"])
    def test_chat_history_grows(self, mock_input, mock_build):
        history_lengths = []

        def capture_invoke(args):
            history_lengths.append(len(args["chat_history"]))
            return {
                "output": json.dumps({
                    "answer": "response",
                    "confidence": "medium",
                    "sources": [],
                    "follow_up_questions": [],
                })
            }

        mock_executor = mock.MagicMock()
        mock_executor.invoke.side_effect = capture_invoke
        mock_build.return_value = mock_executor

        from agent import chat_repl
        chat_repl()

        # First call: empty history; second call: 2 messages (human + ai)
        assert history_lengths == [0, 2]

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["", "exit"])
    def test_empty_input_skipped(self, mock_input, mock_build, capsys):
        mock_executor = mock.MagicMock()
        mock_build.return_value = mock_executor

        from agent import chat_repl
        chat_repl()

        mock_executor.invoke.assert_not_called()

    @mock.patch("agent.build_agent")
    @mock.patch("builtins.input", side_effect=["hello", "exit"])
    def test_error_handling(self, mock_input, mock_build, capsys):
        mock_executor = mock.MagicMock()
        mock_executor.invoke.side_effect = RuntimeError("API timeout")
        mock_build.return_value = mock_executor

        from agent import chat_repl
        chat_repl()

        captured = capsys.readouterr()
        assert "Error: API timeout" in captured.out


# ---------------------------------------------------------------------------
# CLI chat command integration test
# ---------------------------------------------------------------------------

class TestCLIChatCommand:
    def test_main_chat_calls_chat(self):
        with mock.patch("sys.argv", ["cli.py", "chat"]), \
             mock.patch("cli.chat") as mock_chat:
            from cli import main
            main()
            mock_chat.assert_called_once()
