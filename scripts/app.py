# # query.py

# import os
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import CharacterTextSplitter
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import gradio as gr

# # File path for the final dataset
# DATASET_PATH = "../data/final_dataset.txt"

# # Model configurations
# # Use the improved embedding model
# EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# # Use a more capable generation model
# GEN_MODEL_NAME = "google/flan-t5-base"
# TOP_K = 5
# CHUNK_SIZE = 1000       # number of characters per chunk
# CHUNK_OVERLAP = 200     # overlap between chunks

# # 1. Load the final dataset
# with open(DATASET_PATH, "r", encoding="utf-8") as f:
#     final_text = f.read()

# # 2. Split text into manageable chunks for retrieval
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
# chunks = text_splitter.split_text(final_text)

# # 3. Initialize the embedding model
# embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# # 4. Build the FAISS vectorstore from document chunks
# vectorstore = FAISS.from_texts(chunks, embed_model)

# # 5. Set up the HuggingFace LLM (Flan-T5)
# tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
# gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
# hf_pipe = pipeline(
#     "text2text-generation",
#     model=gen_model,
#     tokenizer=tokenizer,
#     return_full_text=False
# )
# llm = HuggingFacePipeline(pipeline=hf_pipe)

# # 6. Create a prompt template for the QA chain with input variable "query"
# prompt_template = """
# Given the following context, answer the question concisely.

# Context:
# {context}

# Question: {query}

# Answer:"""
# prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)

# # 7. Set up the RetrievalQA chain using from_chain_type
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_type="similarity", k=TOP_K),
#     chain_type_kwargs={"prompt": prompt},
#     #input_key="query",  # <-- explicitly tell LangChain what the input variable is
# )
# qa_chain.input_key = "query"


# # 8. Define the answer function for Gradio using .invoke()
# def answer_fn(user_query: str) -> str:
#     try:
#         # use .invoke() to avoid the deprecated .run() API
#         result = qa_chain.invoke({"query": user_query})
#         # handle both dict‐style and string outputs
#         if isinstance(result, dict):
#             response = result.get("result", "")
#         else:
#             response = result or ""
#         response = response.strip()
#         return response or "I don't know"
#     except Exception as e:
#         print("Error in answer_fn:", e)
#         return "I don't know"

# # 9. Launch the Gradio interface (pass share=True to launch, not in Interface constructor)
# if __name__ == "__main__":
#     demo = gr.Interface(
#         fn=answer_fn,
#         inputs=gr.Textbox(lines=2, placeholder="Enter your question..."),
#         outputs="text",
#         title="Angel One Support Chatbot",
#         description="Ask any question about Angel One services using our dataset."
#     )
#     demo.launch(share=True)
# query.py

import os
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# —————————————————————————————
#  Config
# —————————————————————————————

CHUNKS_DIR      = "../data/chunks/"             # folder of chunk_XXXX.txt
FAISS_INDEX     = "../data/faiss_index.index"   # saved Faiss
METADATA_FILE   = "../data/chunks_metadata.txt" # one chunk per line
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
GEN_MODEL       = "google/flan-t5-small"
TOP_K           = 5
SIM_THRESHOLD   = 0.4   # below this, we fallback to "I don't know"
MAX_ANS_TOKENS  = 150

# —————————————————————————————
#  Load resources
# —————————————————————————————

# 1. Load FAISS index
index = faiss.read_index(FAISS_INDEX)

# 2. Load chunk texts
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunk_texts = [line.strip() for line in f if line.strip()]

# 3. Embedding model (must match what you used when indexing)
embedder = SentenceTransformer(EMBED_MODEL)

# 4. Generation pipeline
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen_pipe  = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer,
   
)

# —————————————————————————————
#  Helper: Answer function
# —————————————————————————————

def answer_fn(user_question: str) -> str:
    # 1. Embed the query
    q_emb = embedder.encode([user_question], convert_to_numpy=True)

    # 2. Search FAISS
    D, I = index.search(q_emb, TOP_K)
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        print(f"{rank}. score={score:.4f} → {chunk_texts[idx][:200]!r}")
    if D[0][0] < SIM_THRESHOLD:
        return "I don't know"
    sims, idxs = D[0], I[0]

    # 3. If even the top similarity is too low → I don't know
    if sims[0] < SIM_THRESHOLD:
        return "I don't know"

    # 4. Build context from retrieved chunks
    retrieved = [ chunk_texts[i] for i in idxs ]
    context = "\n\n".join(retrieved)

    # 5. Craft prompt
    prompt = (
    f"Given the following context, answer the question, and explain using common sense\n\n"
    f"CONTEXT:\n{context}\n\n"
    f"QUESTION: {user_question}\n\n"
    "Its fine if you don't know the answer, just say 'I don't know'.\n\n"
    )


    # 6. Generate answer
    out = gen_pipe(prompt, max_length=MAX_ANS_TOKENS)[0]["generated_text"].strip()
    return out or "I don't know"

# —————————————————————————————
#  Launch Gradio
# —————————————————————————————

if __name__ == "__main__":
    demo = gr.Interface(
        fn=answer_fn,
        inputs=gr.Textbox(lines=2, placeholder="Ask about Angel One…"),
        outputs="text",
        title="Angel One Support Chatbot",
        description="Answers based only on the provided support docs."
    )
    demo.launch(share=True)
