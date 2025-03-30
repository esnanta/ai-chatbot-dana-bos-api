import os
import json
import numpy as np
import nltk
import logging
import functools
import torch
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager # Import untuk lifespan

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel

# --- Konstanta ---
# Menggunakan persistent disk Render yang di-mount di /var/data
RENDER_DATA_DIR = "/var/data"
NLTK_DATA_PATH = os.path.join(RENDER_DATA_DIR, "nltk_data")
MODEL_CACHE_PATH = os.path.join(RENDER_DATA_DIR, "model_cache") # Cache model HuggingFace
LOG_FILE_PATH = os.path.join(RENDER_DATA_DIR, "logs", "app.log") # Simpan log di persistent disk

# Diasumsikan knowledge_base ada di root proyek Anda (dari Git)
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunks.json")
EMBEDDING_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunk_embeddings.npy")

# --- Setup Direktori & Logging ---
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True) # Buat direktori logs

# Konfigurasi Logging (log ke console dan file di persistent disk)
logging.basicConfig(
    level=logging.INFO, # Gunakan INFO untuk produksi
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), # Ke console (log Render)
        logging.FileHandler(LOG_FILE_PATH, mode="a") # Ke file di /var/data/logs/
    ]
)
nltk.data.path.append(NLTK_DATA_PATH) # Arahkan NLTK ke persistent disk

# --- Fungsi Helper NLTK ---
def ensure_nltk_data(package: str):
    """Downloads NLTK data if not found to persistent disk."""
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

# --- Global Variables ---
all_chunks: Optional[List[str]] = None
embedder: Optional[SentenceTransformer] = None
cross_encoder_model: Optional[CrossEncoder] = None
chunk_embeddings: Optional[np.ndarray] = None

# --- Fungsi Loading Resources (Dipanggil oleh Lifespan) ---
def load_resources():
    """Loads models, chunks, and embeddings into global variables."""
    global all_chunks, embedder, cross_encoder_model, chunk_embeddings
    logging.info("🔄 Loading models and data...")
    start_time = time.time()

    # --- PERINGATAN KERAS MENGENAI MEMORI ---
    logging.warning("🚨 MEMORY WARNING: Running on 512MB RAM (Render Starter plan).")
    logging.warning("🚨 Loading both SentenceTransformer and CrossEncoder might exceed available memory,")
    logging.warning("🚨 causing restarts or OOM errors. Monitor RAM usage closely in Render dashboard.")
    logging.warning("🚨 Consider upgrading RAM or using only one model if issues occur.")
    # -----------------------------------------

    # Load Chunks (dari direktori proyek)
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

    # Load Sentence Transformer (cache di persistent disk)
    try:
        logging.info(f"💾 Loading Sentence Transformer (cache dir: {MODEL_CACHE_PATH})...")
        embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=MODEL_CACHE_PATH,
            device=device
        )
        logging.info("✅ Sentence Transformer model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Sentence Transformer model: {e}", exc_info=True)
        raise RuntimeError("Failed to load Sentence Transformer") from e

    # Load Cross Encoder (cache di persistent disk)
    try:
        logging.info(f"💾 Loading Cross Encoder (cache dir: {MODEL_CACHE_PATH})...")
        cross_encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", # Pastikan nama ini benar
            cache_folder=MODEL_CACHE_PATH,
            device=device,
            max_length=512
        )
        logging.info("✅ Cross Encoder model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Cross Encoder model: {e}", exc_info=True)
        # Anda bisa memilih untuk tidak menghentikan startup jika CrossEncoder opsional
        # logging.warning("⚠️ Failed to load Cross Encoder, reranking will be skipped.")
        # cross_encoder_model = None # Set ke None jika gagal tapi ingin lanjut
        raise RuntimeError("Failed to load Cross Encoder") from e # Atau hentikan startup

    # Load Embeddings (dari direktori proyek)
    try:
        if not os.path.exists(EMBEDDING_FILE):
             logging.error(f"❌ Embedding file not found at {EMBEDDING_FILE}")
             raise FileNotFoundError(f"Embedding file not found at {EMBEDDING_FILE}")
        chunk_embeddings = np.load(EMBEDDING_FILE)
        logging.info(f"✅ Loaded {chunk_embeddings.shape[0]} embeddings from file.")
        if len(all_chunks) != chunk_embeddings.shape[0]:
             logging.error(f"❌ Mismatch! Chunks ({len(all_chunks)}) vs Embeddings ({chunk_embeddings.shape[0]})")
             raise ValueError("Mismatch between number of chunks and embeddings.")
    except Exception as e:
        logging.error(f"❌ Error loading embeddings from {EMBEDDING_FILE}: {e}", exc_info=True)
        raise RuntimeError("Failed to load embeddings") from e

    end_time = time.time()
    logging.info(f"✅🚀 Models and data successfully loaded in {end_time - start_time:.2f} seconds.")

# --- Lifespan Context Manager (Pengganti on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown logic."""
    logging.info("🚀 Application startup initiated...")
    load_resources() # Panggil fungsi loading kita di sini
    logging.info("✅ Application ready to accept requests.")
    yield
    # Kode di sini dijalankan saat shutdown (jika perlu)
    logging.info("🛑 Application shutdown.")

# --- FastAPI App Initialization dengan Lifespan ---
app = FastAPI(
    title="Question Answering API",
    version="1.0.1", # Naikkan versi
    lifespan=lifespan # Daftarkan lifespan context manager
)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://aichatbot.daraspace.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Fungsi Utility (Cosine Similarity & Finding Chunks) ---
# (Sama seperti sebelumnya, tidak perlu diubah)
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

def find_top_chunks(question_embedding: np.ndarray, top_n: int = 6) -> List[str]:
    if chunk_embeddings is None or all_chunks is None:
        logging.error("❌ Embeddings/chunks not loaded during find_top_chunks.")
        return []
    if not isinstance(question_embedding, np.ndarray):
        logging.error("❌ question_embedding is not np.ndarray.")
        return []
    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
    if not similarities: return []
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [all_chunks[i] for i in top_indices]

# --- Fungsi Caching Embedding Pertanyaan ---
# (Sama seperti sebelumnya)
@functools.lru_cache(maxsize=256)
def get_question_embedding(question: str) -> np.ndarray:
    if embedder is None:
        logging.error("❌ Embedder not loaded for encoding.")
        raise RuntimeError("Embedder not available")
    # logging.debug(f"🧠 Encoding question (cache miss): {question[:50]}...") # Uncomment for debug
    return embedder.encode(question, convert_to_numpy=True)

# --- Fungsi Inti Pemrosesan Pertanyaan ---
def answer_question(question: str, top_n: int = 3) -> str:
    """Processes the question, finds relevant chunks, reranks, and returns the best answer(s)."""
    if embedder is None or chunk_embeddings is None or all_chunks is None:
        logging.critical("❌ CRITICAL: Core resources (Embedder/Chunks/Embeddings) not loaded!")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Core resources missing.")
    # CrossEncoder bisa jadi opsional jika loading gagal (lihat load_resources)
    # if cross_encoder_model is None:
    #    logging.warning("⚠️ CrossEncoder not loaded, skipping reranking step.")

    logging.info(f"🔍 Processing question: {question[:100]}...") # Log sebagian
    start_process_time = time.time()

    try:
        question_embedding = get_question_embedding(question)
    except RuntimeError:
         raise HTTPException(status_code=503, detail="Service temporarily unavailable: Embedder failed.")

    # Retrieval
    candidates = find_top_chunks(question_embedding, top_n=10)
    if not candidates:
        logging.warning("⚠️ No relevant chunks found during initial retrieval.")
        return "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda."

    # Reranking (jika CrossEncoder ada)
    if cross_encoder_model:
        pairs = [(question, chunk) for chunk in candidates]
        try:
            logging.debug(f"📊 Reranking {len(pairs)} pairs...")
            scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
            scored_candidates = sorted(list(zip(scores, candidates)), key=lambda x: x[0], reverse=True)
            top_results = [chunk for score, chunk in scored_candidates[:top_n]]
        except Exception as e:
            logging.error(f"❌ Error during CrossEncoder prediction: {e}", exc_info=True)
            # Fallback ke hasil retrieval jika reranking gagal
            logging.warning("⚠️ Reranking failed, returning top retrieval results.")
            top_results = candidates[:top_n]
    else:
        # Jika CrossEncoder tidak dimuat, langsung gunakan hasil retrieval
        logging.info("ℹ️ Skipping reranking as CrossEncoder is not available.")
        top_results = candidates[:top_n] # Ambil top N dari retrieval

    end_process_time = time.time()
    logging.info(f"✅ Question processed in {end_process_time - start_process_time:.2f} seconds.")

    if not top_results:
         # Fallback terakhir
         return "Maaf, terjadi kesalahan saat memproses jawaban."

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
        answer = answer_question(request.question, top_n=3)
        return {"answer": answer}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.exception("❌ Unexpected error in /ask endpoint:")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/logs", response_model=Dict[str, List[str]])
async def get_logs():
    """Returns the last 50 lines of the application log file from persistent disk."""
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return {"logs": ["Log file not found."]}
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as log_file:
            logs = log_file.readlines()
        return {"logs": [line.strip() for line in logs[-50:]]}
    except Exception as e:
        logging.error("❌ Error reading log file:", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read log file.")

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Basic health check, indicates if models are loaded."""
    status = "ok"
    issues = []
    if embedder is None: issues.append("Embedder not loaded")
    if cross_encoder_model is None: issues.append("CrossEncoder not loaded")
    if chunk_embeddings is None: issues.append("Embeddings not loaded")
    if all_chunks is None: issues.append("Chunks not loaded")

    if issues:
        status = f"degraded ({'; '.join(issues)})"
        logging.warning(f"⚠️ Health check status: {status}")

    return {"status": status, "device": device}

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "Question Answering API is running!"}