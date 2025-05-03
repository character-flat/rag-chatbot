import os
import pdfplumber
import docx
from tqdm import tqdm

INPUT_DIR = "data/insurance_pdfs"
OUTPUT_FILE = "data/insurance_text.txt"


def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_files(input_dir):
    all_text = ""
    for filename in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, filename)
        if filename.lower().endswith(".pdf"):
            all_text += extract_text_from_pdf(path)
        elif filename.lower().endswith(".docx"):
            all_text += extract_text_from_docx(path)
        all_text += "\n"
    return all_text

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Reading files from: {INPUT_DIR}")
    text = extract_text_from_files(INPUT_DIR)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted text written to: {OUTPUT_FILE}")
