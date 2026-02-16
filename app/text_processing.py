import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
#uvicorn app.api:app --reload
def texto_limpio(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_del_texto(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 100
    )

    return splitter.create_documents([text])