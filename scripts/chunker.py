import os
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Settings
INPUT_FILE = "../data/final_dataset.txt"
OUTPUT_DIR = "../data/chunks"
CHUNK_SIZE = 70  # target words per chunk
OVERLAP = 20     # word overlap between chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Use RecursiveCharacterTextSplitter with a word‐based length function
    to enforce a maximum of `chunk_size` words per chunk,
    with `chunk_overlap` words overlapping.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda x: len(x.split()),  # count by words
        separators=["\n\n", "\n", " ", ""],        # split hierarchy
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading from {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Chunking text with RecursiveCharacterTextSplitter...")
    chunks = chunk_text(raw_text)

    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Saving chunks"):
        with open(os.path.join(OUTPUT_DIR, f"chunk_{i:04d}.txt"), "w", encoding="utf-8") as out_f:
            out_f.write(chunk)

    print(f"Done! {len(chunks)} chunks saved to {OUTPUT_DIR}")
