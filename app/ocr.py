import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_para_imagen(image_path):
    imagen = Image.open(image_path)
    texto = pytesseract.image_to_string(imagen, lang = "spa")
    return texto

