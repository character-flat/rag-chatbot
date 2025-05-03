import os
import glob

CLEANED_DIR      = "/workspaces/rag-chatbot/data/cleaned_pages"
INSURANCE_FILE   = "/workspaces/rag-chatbot/data/insurance_text.txt"
COMBINED_CLEANED = "/workspaces/rag-chatbot/data/combined_cleaned.txt"
FINAL_OUTPUT     = "/workspaces/rag-chatbot/data/final_dataset.txt"

def combine_cleaned_pages():
    txt_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*.txt")))
    with open(COMBINED_CLEANED, "w", encoding="utf-8") as out:
        for path in txt_files:
            out.write(f"# ---- {os.path.basename(path)} ----\n")
            with open(path, "r", encoding="utf-8") as inp:
                out.write(inp.read().strip() + "\n\n")
    print(f"✅ Combined {len(txt_files)} cleaned files into {COMBINED_CLEANED}")

def build_final_dataset():
    # read insurance_text.py
    with open(INSURANCE_FILE, "r", encoding="utf-8") as ins:
        insurance = ins.read().rstrip()

    # read combined cleaned pages
    with open(COMBINED_CLEANED, "r", encoding="utf-8") as cmb:
        scraped = cmb.read().rstrip()

    # write final merged file
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as out:
        out.write(insurance + "\n\n")
        out.write(scraped + "\n")
    print(f"✅ Final dataset written to {FINAL_OUTPUT}")

if __name__ == "__main__":
    combine_cleaned_pages()
    build_final_dataset()