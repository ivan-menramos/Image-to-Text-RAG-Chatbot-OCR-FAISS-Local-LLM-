# Sistema RAG con OCR para Imágenes

Sistema de preguntas y respuestas sobre imágenes mediante extracción de texto (OCR) y generación aumentada por recuperación (RAG).

## Descripción del proyecto


1. Subir una imagen (JPG, PNG, etc.)
2. Extraer texto usando OCR (Tesseract)
3. Dividir el texto en fragmentos
4. Generar embeddings y almacenarlos en FAISS
5. Usar google/flan-t5-base para responder preguntas basadas en el contenido extraído

---

## Arquitectura

Usuario -> FastAPI -> OCR -> Chunking -> Embeddings -> FAISS -> Retriever -> Flan-T5 -> Respuesta

---

## Tecnologías

- Python
- FastAPI
- pytesseract
- Pillow
- LangChain
- FAISS
- google/flan-t5-base

---

## Características

- Extracción de texto desde imágenes
- Modelo seq2seq (Flan-T5)
- RAG totalmente local
- Sistema híbrido OCR + NLP
- API REST

---

## Aprendizajes

- Integración de OCR en pipelines de IA
- Diseño de sistemas híbridos Vision + NLP
- Generación controlada con modelos encoder-decoder
