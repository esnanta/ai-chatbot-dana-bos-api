import os
import json
import faiss
import numpy as np
import uvicorn
import nltk
import logging
import functools

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel
import time

nltk.download('punkt')
nltk.download('punkt_tab')

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)

app = FastAPI()

# Konfigurasi CORS
origins = ["http://localhost", "http://localhost:80", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# ===============================
# 1. KONSTANTA & INISIALISASI GLOBAL
# ===============================

BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
EMBEDDING_DIMENSION = 384
EMBEDDING_FILE = os.path.join(BASE_DIR, "chunk_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")
NPROBE = 5

# ===============================
# 2. MUAT DATA & MODEL (CACHING)
# ===============================
logging.info("Loading models and data...")
start_time = time.time()

try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        ALL_CHUNKS = json.load(f)
        logging.info(f"✅ Loaded {len(ALL_CHUNKS)} chunks.")

    EMBEDDER = SentenceTransformer("paraphrase-MiniLM-L3-v2", cache_folder="./model_cache")
    logging.info("✅ SentenceTransformer model loaded.")

    CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6", cache_folder="./model_cache")
    logging.info("✅ CrossEncoder model loaded.")

    if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
        CHUNK_EMBEDDINGS = np.load(EMBEDDING_FILE)
        INDEX_FAISS = faiss.read_index(FAISS_INDEX_FILE)
        INDEX_FAISS.nprobe = NPROBE
        logging.info(f"✅ Loaded FAISS index with {len(CHUNK_EMBEDDINGS)} embeddings.")
    else:
        raise RuntimeError("❌ FAISS index file is missing!")

    end_time = time.time()
    logging.info(f"✅ Models and data loaded in {end_time - start_time:.2f} seconds.")

except Exception as e:
    logging.exception("❌ Error during model loading:")
    raise

# ===============================
# 3. CACHING & UTILITAS
# ===============================
@functools.lru_cache(maxsize=100)
def cache_question_embedding(question: str):
    return EMBEDDER.encode([question], convert_to_numpy=True)

@functools.lru_cache(maxsize=100)
def cache_faiss_search(question: str):
    question_embedding = cache_question_embedding(question)
    D, I = INDEX_FAISS.search(question_embedding, 6)
    return [ALL_CHUNKS[i] for i in I[0]]

def post_process_answer(answer):
    sentences = sent_tokenize(answer)
    unique_sentences = list(dict.fromkeys(sentences))
    return "<br><br>".join([f"* {sentence}" for sentence in unique_sentences if len(sentence.strip()) > 10])

def answer_question(question: str, top_n: int = 3) -> str:
    try:
        logging.info(f"🔍 Processing question: {question}")
        candidates = cache_faiss_search(question)
        pairs = [(question, chunk) for chunk in candidates]
        scores = CROSS_ENCODER_MODEL.predict(pairs, batch_size=4)
        top_indices = np.argsort(scores)[::-1][:top_n]
        final_answers = "\n".join([candidates[i] for i in top_indices])
        logging.info("✅ Answer generated successfully.")
        return final_answers
    except Exception as e:
        logging.exception("❌ Error in answer_question:")
        return "Error: Terjadi kesalahan dalam memproses pertanyaan."

# ===============================
# 4. FASTAPI ENDPOINTS
# ===============================
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_chatbot(request: QuestionRequest):
    try:
        logging.info(f"📩 Received API request: {request.question}")
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")

        raw_answer = answer_question(request.question, top_n=3)
        processed_answer = raw_answer.replace("\n", "<br><br>")
        return {"answer": processed_answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("❌ Error processing request:")
        raise HTTPException(status_code=500, detail="An error occurred processing the request")

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# ===============================
# 5. MENJALANKAN SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logging.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
