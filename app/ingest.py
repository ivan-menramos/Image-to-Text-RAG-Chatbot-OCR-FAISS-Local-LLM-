import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.ocr import ocr_para_imagen
from app.text_processing import texto_limpio, split_del_texto

def extraer_texto(file_path):
    
    if file_path.endswith((".png", ".jpg",".jpeg")):
        return ocr_para_imagen(file_path)
    
    else:
        raise ValueError("Formato no soportado")
    
def construir_vectorstore(file_path):
    texto_crudo = extraer_texto(file_path)
    texto_procesado = texto_limpio(texto_crudo)
    documentos = split_del_texto(texto_procesado)

    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documentos, embeddings)

    vectorstore.save_local("app/vectorstore")

    return vectorstore