import sys
import fitz

def extract_pdf(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

extract_pdf(r"c:\Users\gfilg\Documentos-Real\UZH\bayesian\exercise\FS2026_homework2.pdf", "ex2.txt")
extract_pdf(r"c:\Users\gfilg\Documentos-Real\UZH\bayesian\past-exercise\homework 1 causal inference_comments.pdf", "ex1.txt")
