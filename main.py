import os
import json
import faiss
import numpy as np
import uvicorn
import nltk
import logging
import functools
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel
import time

# Konstanta
NLTK_DATA_PATH = "/var/data/nltk_data"
MODEL_DATA_PATH = "/var/data"
BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
EMBEDDING_DIMENSION = 384
EMBEDDING_FILE = os.path.join(BASE_DIR, "chunk_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")
NPROBE = 5

# Pastikan direktori ada
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_DATA_PATH, exist_ok=True)

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)

nltk.data.path.append(NLTK_DATA_PATH)

# Cek dan download NLTK dataset jika belum ada
def ensure_nltk_data(package: str):
    try:
        nltk.data.find(f"tokenizers/{package}")
        logging.info(f"✅ NLTK dataset '{package}' already exists.")
    except LookupError:
        logging.warning(f"⚠️ NLTK dataset '{package}' not found. Downloading...")
        nltk.download(package, download_dir=NLTK_DATA_PATH)

ensure_nltk_data('punkt')
ensure_nltk_data('punkt_tab')

# Cek GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Using device: {device}")

# Muat Model
logging.info("Loading models and data...")
start_time = time.time()

try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        ALL_CHUNKS = json.load(f)
        logging.info(f"✅ Loaded {len(ALL_CHUNKS)} chunks.")

    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=MODEL_DATA_PATH, device=device)
    CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", cache_folder=MODEL_DATA_PATH,
                                       device=device)

    logging.info(f"📂 Cache folder contents: {os.listdir(MODEL_DATA_PATH)}")
    logging.info("✅ Models loaded.")

    if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
        CHUNK_EMBEDDINGS = np.load(EMBEDDING_FILE)
        INDEX_FAISS = faiss.read_index(FAISS_INDEX_FILE)

        if CHUNK_EMBEDDINGS.shape[0] != INDEX_FAISS.ntotal:
            logging.error("❌ FAISS index mismatch!")
            raise RuntimeError("FAISS index and embedding file are not in sync!")

        INDEX_FAISS.nprobe = NPROBE
        logging.info(f"✅ Loaded FAISS index with {len(CHUNK_EMBEDDINGS)} embeddings.")
    else:
        raise RuntimeError("❌ FAISS index file is missing!")

    end_time = time.time()
    logging.info(f"✅ Models and data loaded in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logging.exception("❌ Error during model loading:")
    raise

# FastAPI app
app = FastAPI()
origins = ["http://localhost", "http://localhost:8000", "https://aichatbot.daraspace.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# Caching & Utilitas
@functools.lru_cache(maxsize=100)
def cache_question_embedding(question: str):
    return EMBEDDER.encode([question], convert_to_numpy=True)


@functools.lru_cache(maxsize=100)
def cache_faiss_search(question: str):
    question_embedding = cache_question_embedding(question)
    D, I = INDEX_FAISS.search(question_embedding, 6)
    return [ALL_CHUNKS[i] for i in I[0] if i >= 0]


def answer_question(question: str, top_n: int = 3) -> str:
    logging.info(f"🔍 Processing question: {question}")
    candidates = cache_faiss_search(question)
    if not candidates:
        logging.warning("⚠️ No relevant chunks found for the question.")
        return "Maaf, saya tidak dapat menemukan jawaban yang sesuai."

    pairs = [(question, chunk) for chunk in candidates]
    scores = CROSS_ENCODER_MODEL.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return "<br><br>".join([candidates[i] for i in top_indices])


# API Endpoint
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_chatbot(request: QuestionRequest):
    try:
        logging.info(f"📩 Received API request: {request.question}")
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")
        return {"answer": answer_question(request.question, top_n=3)}
    except Exception as e:
        logging.exception("❌ Error processing request:")
        raise HTTPException(status_code=500, detail="An error occurred processing the request")

@app.get("/logs")
async def get_logs():
    try:
        with open("app.log", "r", encoding="utf-8") as log_file:
            logs = log_file.readlines()
        return {"logs": logs[-50:]}  # Ambil 50 baris terakhir untuk menghindari terlalu banyak data
    except Exception as e:
        logging.exception("❌ Error reading log file:")
        raise HTTPException(status_code=500, detail="An error occurred while reading the log file")

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": device}


@app.get("/")
def read_root():
    return {"message": "API is running!"}


# DISABLE THIS FOR RENDER
#if __name__ == "__main__":
#    port = int(os.getenv("PORT", 8000))
#    logging.info(f"🚀 Starting server on port {port}")
#   uvicorn.run(app, host="0.0.0.0", port=port)
