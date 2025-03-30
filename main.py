import os
import json
import numpy as np
import nltk
import logging
import functools
import torch
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# HANYA import SentenceTransformer
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# --- Konstanta ---
RENDER_DATA_DIR = "/var/data"
NLTK_DATA_PATH = os.path.join(RENDER_DATA_DIR, "nltk_data")
MODEL_CACHE_PATH = os.path.join(RENDER_DATA_DIR, "model_cache")
LOG_FILE_PATH = os.path.join(RENDER_DATA_DIR, "logs", "app.log")
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunks.json")
EMBEDDING_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunk_embeddings.npy")

# --- Setup Direktori & Logging ---
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_PATH, mode="a")
    ]
)
nltk.data.path.append(NLTK_DATA_PATH)

# --- Fungsi Helper NLTK ---
def ensure_nltk_data(package: str):
    try:
        nltk.find(f"tokenizers/{package}")
        logging.info(f"✅ NLTK dataset '{package}' found in {NLTK_DATA_PATH}.")
    except LookupError:
        logging.warning(f"⚠️ NLTK dataset '{package}' not found. Downloading to {NLTK_DATA_PATH}...")
        try:
            nltk.download(package, download_dir=NLTK_DATA_PATH)
            logging.info(f"✅ Successfully downloaded NLTK dataset '{package}' to persistent disk.")
        except Exception as e:
            logging.error(f"❌ Failed to download NLTK dataset '{package}': {e}", exc_info=True)
            raise RuntimeError(f"Could not download NLTK data: {package}") from e

# --- Pengecekan Awal ---
ensure_nltk_data('punkt')
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Using device: {device}")

# --- Global Variables (TANPA CrossEncoder) ---
all_chunks: Optional[List[str]] = None
embedder: Optional[SentenceTransformer] = None
# Hapus cross_encoder_model
chunk_embeddings: Optional[np.ndarray] = None

# --- Fungsi Loading Resources (TANPA CrossEncoder) ---
def load_resources():
    """Loads Sentence Transformer, chunks, and embeddings into global variables."""
    # Hanya perlu global untuk yang dimuat
    global all_chunks, embedder, chunk_embeddings
    logging.info("🔄 Loading models and data...")
    start_time = time.time()

    # Sesuaikan Peringatan Memori
    logging.warning("🚨 MEMORY INFO: Running on 512MB RAM (Render Starter plan).")
    logging.warning("🚨 Loading Sentence Transformer model. Monitor RAM usage closely.")
    logging.warning("🚨 Upgrade RAM if OOM errors still occur with this model.")

    # Load Chunks
    try:
        if not os.path.exists(CHUNKS_FILE):
            logging.error(f"❌ Chunks file not found at {CHUNKS_FILE}")
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        logging.info(f"✅ Loaded {len(all_chunks)} chunks.")
    except Exception as e:
        logging.error(f"❌ Error loading chunks from {CHUNKS_FILE}: {e}", exc_info=True)
        raise RuntimeError("Failed to load chunks") from e

    # Load Sentence Transformer (Gunakan model yang lebih kecil)
    # PASTIKAN embedding Anda dibuat dengan model INI!
    selected_model = "paraphrase-MiniLM-L3-v2"
    logging.info(f"💾 Loading Sentence Transformer model: '{selected_model}' (cache dir: {MODEL_CACHE_PATH})...")
    try:
        embedder = SentenceTransformer(
            selected_model,
            cache_folder=MODEL_CACHE_PATH,
            device=device
        )
        logging.info("✅ Sentence Transformer model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Sentence Transformer model '{selected_model}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Sentence Transformer: {selected_model}") from e

    # HAPUS BLOK LOADING CROSS ENCODER SELURUHNYA
    # try:
    #     ... (kode loading cross encoder dihapus) ...
    # except Exception as e:
    #     ...

    # Load Embeddings
    # PASTIKAN file ini dibuat menggunakan model 'paraphrase-MiniLM-L3-v2'
    logging.info(f"💾 Loading embeddings from {EMBEDDING_FILE} (ensure they match model '{selected_model}')...")
    try:
        if not os.path.exists(EMBEDDING_FILE):
             logging.error(f"❌ Embedding file not found at {EMBEDDING_FILE}")
             raise FileNotFoundError(f"Embedding file not found at {EMBEDDING_FILE}")
        chunk_embeddings = np.load(EMBEDDING_FILE)
        logging.info(f"✅ Loaded {chunk_embeddings.shape[0]} embeddings from file.")
        if len(all_chunks) != chunk_embeddings.shape[0]:
             logging.error(f"❌ Mismatch! Chunks ({len(all_chunks)}) vs Embeddings ({chunk_embeddings.shape[0]})")
             raise ValueError("Mismatch between number of chunks and embeddings.")
        # Cek dimensi embedding (opsional tapi bagus)
        expected_dim = embedder.get_sentence_embedding_dimension()
        if chunk_embeddings.shape[1] != expected_dim:
            logging.error(f"❌ Embedding dimension mismatch! Expected {expected_dim}, got {chunk_embeddings.shape[1]}. Regenerate embeddings with model '{selected_model}'.")
            raise ValueError("Embedding dimension mismatch.")

    except Exception as e:
        logging.error(f"❌ Error loading embeddings from {EMBEDDING_FILE}: {e}", exc_info=True)
        raise RuntimeError("Failed to load embeddings") from e

    end_time = time.time()
    logging.info(f"✅🚀 Models and data successfully loaded in {end_time - start_time:.2f} seconds.")

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Application startup initiated...")
    load_resources()
    logging.info("✅ Application ready to accept requests.")
    yield
    logging.info("🛑 Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Question Answering API (Embedder Only)", # Update title
    version="1.1.0", # Naikkan versi
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost", "http://localhost:8000", "http://localhost:3000",
    "https://aichatbot.daraspace.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

# --- Fungsi Utility (Cosine Similarity & Finding Chunks) ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

def find_top_chunks(question_embedding: np.ndarray, top_n: int = 6) -> List[str]:
    """Finds top N chunks based purely on cosine similarity."""
    if chunk_embeddings is None or all_chunks is None:
        logging.error("❌ Embeddings/chunks not loaded during find_top_chunks.")
        return []
    if not isinstance(question_embedding, np.ndarray):
        logging.error("❌ question_embedding is not np.ndarray.")
        return []

    similarities = np.array([cosine_similarity(question_embedding, emb) for emb in chunk_embeddings])

    if len(similarities) == 0: return []

    # Ambil top_n index dari yang terbesar
    # np.argpartition bisa lebih cepat untuk N besar, tapi argsort mudah dibaca
    # Untuk mengambil N terbesar: indices = np.argsort(similarities)[-top_n:]
    # Untuk mengambil N terbesar DAN mengurutkannya dari terbesar ke kecil:
    top_indices = np.argsort(similarities)[::-1][:top_n] # Sort descending, take top N

    return [all_chunks[i] for i in top_indices]

# --- Fungsi Caching Embedding Pertanyaan ---
@functools.lru_cache(maxsize=256)
def get_question_embedding(question: str) -> np.ndarray:
    if embedder is None:
        logging.error("❌ Embedder not loaded for encoding.")
        raise RuntimeError("Embedder not available")
    return embedder.encode(question, convert_to_numpy=True)

# --- Fungsi Inti Pemrosesan Pertanyaan (TANPA Reranking) ---
def answer_question(question: str, top_n: int = 3) -> str:
    """Processes question using only Sentence Transformer and Cosine Similarity."""
    if embedder is None or chunk_embeddings is None or all_chunks is None:
        logging.critical("❌ CRITICAL: Core resources (Embedder/Chunks/Embeddings) not loaded!")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Core resources missing.")

    logging.info(f"🔍 Processing question (Embedder Only): {question[:100]}...")
    start_process_time = time.time()

    try:
        question_embedding = get_question_embedding(question)
    except RuntimeError:
         raise HTTPException(status_code=503, detail="Service temporarily unavailable: Embedder failed.")

    # Retrieval (langsung ambil top_n yang dibutuhkan)
    # top_n+2 untuk sedikit cadangan jika ada duplikat, tapi top_n harusnya cukup
    # Kita akan ambil sejumlah `top_n` langsung dari find_top_chunks
    logging.debug(f"📊 Finding top {top_n} chunks using cosine similarity...")
    top_results = find_top_chunks(question_embedding, top_n=top_n)

    if not top_results:
        logging.warning("⚠️ No relevant chunks found.")
        return "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda."

    # HAPUS BLOK RERANKING (if cross_encoder_model: ... else: ...)
    # Hasil dari find_top_chunks sudah merupakan hasil akhir

    end_process_time = time.time()
    logging.info(f"✅ Question processed in {end_process_time - start_process_time:.2f} seconds (Embedder Only).")

    return "<br><br>".join(top_results)

# --- API Endpoints ---
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", response_model=Dict[str, str])
async def ask_chatbot(request: QuestionRequest):
    try:
        logging.info(f"📩 Received API request: {request.question[:100]}...")
        if not request.question or not request.question.strip():
            logging.warning("⚠️ Received empty question.")
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        # Panggil fungsi answer_question yang sudah disederhanakan
        answer = answer_question(request.question, top_n=3)
        return {"answer": answer}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.exception("❌ Unexpected error in /ask endpoint:")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/logs", response_model=Dict[str, List[str]])
async def get_logs():
    try:
        if not os.path.exists(LOG_FILE_PATH): return {"logs": ["Log file not found."]}
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as log_file:
            logs = log_file.readlines()
        return {"logs": [line.strip() for line in logs[-50:]]}
    except Exception as e:
        logging.error("❌ Error reading log file:", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read log file.")

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check (checks Embedder, Chunks, Embeddings)."""
    status = "ok"
    issues = []
    if embedder is None: issues.append("Embedder not loaded")
    # Hapus pengecekan CrossEncoder
    if chunk_embeddings is None: issues.append("Embeddings not loaded")
    if all_chunks is None: issues.append("Chunks not loaded")

    if issues:
        status = f"degraded ({'; '.join(issues)})"
        logging.warning(f"⚠️ Health check status: {status}")

    return {"status": status, "device": device}

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "Question Answering API (Embedder Only) is running!"} # Update pesan