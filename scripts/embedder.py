import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNKS_DIR = "../data/chunks"
# Use a high-quality Instructor model for embeddings
MODEL_NAME = "hkunlp/instructor-large"
INDEX_OUTPUT = "../data/faiss_index.index"
METADATA_OUTPUT = "../data/chunks_metadata.txt"

# 1. Load model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# 2. Read all chunk files
print(f"Reading chunks from {CHUNKS_DIR}")
chunk_files = sorted(f for f in os.listdir(CHUNKS_DIR) if f.startswith("chunk_") and f.endswith(".txt"))

chunks = []
for fname in chunk_files:
    with open(os.path.join(CHUNKS_DIR, fname), "r", encoding="utf-8") as f:
        chunk_text = f.read().strip()
        if chunk_text:
            chunks.append(chunk_text)

print(f"Found {len(chunks)} chunks.")

# 3. Preprocess: add instruction prefix for the Instructor model
instruction = "Represent the following text for retrieval:"
processed_chunks = [f"{instruction}\n\n{c}" for c in chunks]

# 4. Embed
print("Generating embeddings...")
embeddings = model.encode(processed_chunks, show_progress_bar=True, convert_to_numpy=True)

# 5. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Save index
faiss.write_index(index, INDEX_OUTPUT)
print(f"FAISS index saved to {INDEX_OUTPUT}")

# 7. Save metadata
with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")
print(f"Chunk metadata saved to {METADATA_OUTPUT}")
