import os
from langchain_community.vectorstores import FAISS
from config import VECTOR_STORE, VECTORSTORE_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME
from embeddings import get_embeddings


def create_vectorstore(docs):
    emb = get_embeddings()

    if VECTOR_STORE == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore.from_documents(
            docs, emb,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY,
        )

    vs = FAISS.from_documents(docs, emb)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    print(f"FAISS index saved to {VECTORSTORE_DIR}/")
    return vs


def load_vectorstore():
    emb = get_embeddings()

    if VECTOR_STORE == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=emb,
            pinecone_api_key=PINECONE_API_KEY,
        )

    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        raise FileNotFoundError("No FAISS index found. Run 'python cli.py ingest' first.")
    return FAISS.load_local(VECTORSTORE_DIR, emb, allow_dangerous_deserialization=True)


def get_retriever(k=4):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
