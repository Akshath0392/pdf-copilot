import argparse
import sys


def ingest():
    from loader import load_and_split
    from store import create_vectorstore

    docs = load_and_split()
    create_vectorstore(docs)
    print("Ingestion complete.")


def ask(question: str):
    from chain import get_qa_chain

    qa = get_qa_chain()
    result = qa.invoke({"query": question})

    print("\nAnswer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  - {src}  (page {page})")


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ingest", help="Load PDFs, chunk, embed, and store")

    ask_parser = sub.add_parser("ask", help="Ask a question against ingested documents")
    ask_parser.add_argument("question", help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        ask(args.question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
