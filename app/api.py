from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
import shutil

from app.ingest import construir_vectorstore
from app.rag_chain import construir_rag

app = FastAPI()

qa_chain = None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    construir_vectorstore(file_path)

    global qa_chain
    qa_chain = construir_rag()

    return {"message": "Documento procesado correctamente"}


@app.post("/ask", response_class=PlainTextResponse)
async def ask_question(question: str):

    response = qa_chain.invoke({"query": question})

    return response["result"]
