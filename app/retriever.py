from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def construir_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "app/vectorstore",
        embeddings,
        allow_dangerous_deserialization = True
    )

    return vectorstore.as_retriever(search_kwargs = {"k" : 3})