from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
#from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch
from app.retriever import construir_retriever



model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 200,
    temperature = 0.3
    )

llm = HuggingFacePipeline(pipeline = pipe)

def construir_rag():
    
    retriever = construir_retriever()
    prompt_template = """
Solamente responde usando el contexto. 
Si no se encuentra en el contexto, responde:
No encontré la información requerida

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

    PROMPT = PromptTemplate(
        template = prompt_template,
        input_variables = ["context","question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        chain_type_kwargs = {"prompt": PROMPT},
        return_source_documents = True
    )

    return qa_chain