from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
)
from store import get_retriever

_PROMPT_TEMPLATE = """Use the following context to answer the question. If the answer
is not contained in the context, say "I don't have enough information to answer that."
Do not fabricate information.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm():
    if LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
            "Supported providers: claude, openai, gemini"
        )


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever(k=4)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
