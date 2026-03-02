import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel

from chain import get_llm
from store import get_retriever


# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    document: str
    page: int


class AgentResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[SourceInfo] = []
    follow_up_questions: list[str] = []


# ---------------------------------------------------------------------------
# RAG search tool
# ---------------------------------------------------------------------------

@tool
def search_compliance_docs(query: str) -> str:
    """Search the ingested compliance documents for information relevant to the query."""
    retriever = get_retriever(k=4)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an RBI compliance assistant. Use the search_compliance_docs tool to find relevant information before answering.

Always respond with valid JSON matching this schema:
{{
  "answer": "your detailed answer",
  "confidence": "high" | "medium" | "low",
  "sources": [{{"document": "filename", "page": 1}}],
  "follow_up_questions": ["question 1", "question 2"]
}}

Confidence guidelines:
- "high": answer is directly supported by retrieved documents
- "medium": answer is partially supported or inferred from documents
- "low": answer is based on general knowledge, not found in documents"""


def build_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [search_compliance_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        handle_parsing_errors=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(raw_output: str) -> AgentResponse:
    text = raw_output.strip()

    # Try 1: direct JSON parse
    try:
        return AgentResponse(**json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass

    # Try 2: extract from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return AgentResponse(**json.loads(fence_match.group(1)))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3: find first {...} block in text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return AgentResponse(**json.loads(brace_match.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: wrap raw text as low-confidence answer
    return AgentResponse(
        answer=text,
        confidence="low",
        sources=[],
        follow_up_questions=[],
    )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def chat_repl():
    print("RBI Compliance Assistant (type 'exit' or 'quit' to stop)")
    print("-" * 50)

    agent_executor = build_agent()
    chat_history: list = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = agent_executor.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            raw = result["output"]
            response = parse_response(raw)
            print(f"\nAssistant: {response.model_dump_json(indent=2)}")
        except Exception as e:
            raw = f"Error: {e}"
            response = AgentResponse(
                answer=raw,
                confidence="low",
                sources=[],
                follow_up_questions=[],
            )
            print(f"\nAssistant: {response.model_dump_json(indent=2)}")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=raw))
