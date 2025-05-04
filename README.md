
# 🧠 RAG Chatbot (FOSS-Only Edition)

This is a lightweight, fully open-source Retrieval-Augmented Generation (RAG) chatbot that answers questions from long documents (e.g. PDFs, scraped websites, etc.). It uses:

- ✅ Open-source embedding models (`sentence-transformers`)
- ✅ FAISS for vector similarity search
- ✅ `google/flan-t5-small` for answer generation
- ✅ Local preprocessing with chunking and cleaning
- ✅ Gradio frontend (optional)

---

## 🧩 Architecture

```

User Question
│
▼
\[ Embedding Model ]
│
▼
\[ FAISS Vector DB ] ← Preprocessed Chunks
│
Top-k Most Similar Chunks
│
▼
\[ FLAN-T5-Small ]
│
▼
Generated Answer

```

---

## 📚 Data Source

The input data used in this project comes from **insurance-related PDFs and scraped coverage summaries**. These documents contain:

- Summaries of benefits and coverage
- Tables for deductibles, co-pays, and network benefits
- Legal and regulatory disclaimers

These were cleaned and converted to plain text in `data/final_dataset.txt`.

> 📌 Note: The quality of responses depends heavily on how relevant and clean this source data is. Garbage in, garbage out.

---

## ✅ Advantages

- **Fully Open-Source**: No APIs, no vendor lock-in.
- **Lightweight**: Runs on CPU or small GPU; suitable for local or Hugging Face Spaces deployment.
- **Customizable**: Easily change models, chunk sizes, or retrievers.
- **Privacy-Preserving**: No user data or documents leave your system.

---

## ❌ Limitations

- **Small Embedding Model**: Struggles with nuanced or domain-specific queries. Poor similarity scores can return irrelevant chunks.
- **Small Generator (FLAN-T5-Small)**: May hallucinate or give vague answers when context is weak.
- **No Reranker Yet**: Retrieval is purely based on cosine similarity; wrong top-k results degrade answer quality fast.
- **Bad Data = Bad Answers**: If the underlying dataset is off-topic, the bot cannot magically "understand" your question.

---

## 📂 Folder Structure

```

.
├── data/
│   ├── final\_dataset.txt        # Raw input text (from insurance PDFs)
│   └── chunks/                  # Preprocessed chunks
├── embed.py                     # Generates embeddings & builds FAISS index
├── query.py                     # Answers user questions
├── chunker.py                   # Splits large documents into overlapping chunks
├── app.py                       # Gradio frontend (optional)
└── README.md

````

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Chunk your document

```bash
python chunker.py
```

### 3. Embed & build vector index

```bash
python embed.py
```

### 4. Ask questions

```bash
python query.py
```

### 5. (Optional) Launch Gradio app

```bash
python app.py
```

---

## 🧪 Tips for Better Performance

* Use **smaller chunk sizes** (e.g., 80–120 words) with overlap (e.g., 30 words).
* Filter out irrelevant content like glossaries, URLs, or legalese.
* Match your dataset to your domain (finance dataset for finance Qs, etc.).
* Try better embedding models like `bge-small-en-v1.5` or `e5-small-v2`.

---

## 🛠️ Future Improvements

* Add a reranker model to improve retrieval relevance.
* Switch to `flan-t5-base` or `Mistral` for stronger generation.
* Add prompt templates with clearer instructions.
* Multi-document ingestion and metadata filtering.

---

## ✊ License

MIT. Built with 100% open-source tools and vibes.

```


