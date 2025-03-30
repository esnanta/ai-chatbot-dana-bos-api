import os
import json
import numpy as np
import nltk
import logging
import functools # Untuk lru_cache
import torch
import time
import re # Untuk post_process_answer

from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Import SentenceTransformer dan util
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize # Untuk post_process_answer
from pydantic import BaseModel

# --- Konstanta ---
RENDER_DATA_DIR = os.environ.get("RENDER_DATA_DIR", "/var/data") # Gunakan env var jika diset
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
        logging.StreamHandler(), # Ke console (log Render)
        logging.FileHandler(LOG_FILE_PATH, mode="a") # Ke file di persistent disk
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
            nltk.download(package, download_dir=NLTK_DATA_PATH, quiet=True)
            logging.info(f"✅ Successfully downloaded NLTK dataset '{package}'.")
        except Exception as e:
            logging.error(f"❌ Failed to download NLTK dataset '{package}': {e}", exc_info=True)
            raise RuntimeError(f"Could not download NLTK data: {package}") from e

# --- Pengecekan Awal ---
ensure_nltk_data('punkt') # Untuk sent_tokenize
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Using device: {device}")

# --- Global Variables (Hanya Embedder) ---
all_chunks: Optional[List[str]] = None
embedder: Optional[SentenceTransformer] = None
chunk_embeddings: Optional[np.ndarray] = None

# --- Fungsi Loading Resources (Hanya Embedder) ---
def load_resources():
    """Loads Sentence Transformer, chunks, and embeddings."""
    global all_chunks, embedder, chunk_embeddings
    logging.info("🔄 Loading models and data for Embedder-Only pipeline...")
    start_time = time.time()

    # --- Pilih Model Embedder ---
    # Pastikan ini SAMA dengan model yang digunakan untuk membuat EMBEDDING_FILE
    # selected_model = "paraphrase-MiniLM-L3-v2" # Opsi kecil
    selected_model = "all-MiniLM-L6-v2"         # Opsi umum
    # selected_model = "multi-qa-MiniLM-L6-cos-v1" # Opsi Q&A
    logging.info(f"Selected embedder model: {selected_model}")
    logging.warning(f"🚨 ENSURE '{EMBEDDING_FILE}' was generated using '{selected_model}'!")

    # Load Chunks
    try:
        if not os.path.exists(CHUNKS_FILE):
            logging.error(f"❌ Chunks file not found at {CHUNKS_FILE}")
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        logging.info(f"✅ Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}.")
    except Exception as e:
        logging.error(f"❌ Error loading chunks: {e}", exc_info=True)
        raise RuntimeError("Failed to load chunks") from e

    # Load Sentence Transformer
    try:
        logging.info(f"💾 Loading Sentence Transformer model: '{selected_model}' (cache: {MODEL_CACHE_PATH})...")
        embedder = SentenceTransformer(
            selected_model,
            cache_folder=MODEL_CACHE_PATH,
            device=device
        )
        logging.info("✅ Sentence Transformer model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Sentence Transformer model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Sentence Transformer: {selected_model}") from e

    # Dapatkan dimensi embedding yang diharapkan dari model
    try:
        expected_dim = embedder.get_sentence_embedding_dimension()
        logging.info(f"Model embedding dimension: {expected_dim}")
    except Exception as e:
        logging.error(f"Could not get embedding dimension from model: {e}")
        expected_dim = None

    # Load Embeddings
    try:
        if not os.path.exists(EMBEDDING_FILE):
             logging.error(f"❌ Embedding file not found at {EMBEDDING_FILE}")
             raise FileNotFoundError(f"Embedding file not found at {EMBEDDING_FILE}. Generate it first matching model '{selected_model}'.")
        logging.info(f"💾 Loading embeddings from {EMBEDDING_FILE}...")
        chunk_embeddings = np.load(EMBEDDING_FILE)
        logging.info(f"✅ Loaded embeddings with shape {chunk_embeddings.shape}.")

        # --- Validasi Krusial ---
        if len(all_chunks) != chunk_embeddings.shape[0]:
             logging.error(f"❌ Mismatch! Chunks count ({len(all_chunks)}) != Embeddings count ({chunk_embeddings.shape[0]}). Check {CHUNKS_FILE} and {EMBEDDING_FILE}.")
             raise ValueError("Mismatch between number of chunks and embeddings.")
        if expected_dim is not None and chunk_embeddings.shape[1] != expected_dim:
            logging.error(f"❌ Embedding dimension mismatch! Expected {expected_dim} (from model '{selected_model}'), but file has {chunk_embeddings.shape[1]}. Regenerate '{EMBEDDING_FILE}'.")
            raise ValueError("Embedding dimension mismatch.")
        # --- Akhir Validasi ---

    except Exception as e:
        logging.error(f"❌ Error loading embeddings: {e}", exc_info=True)
        raise RuntimeError("Failed to load embeddings") from e

    end_time = time.time()
    logging.info(f"✅🚀 Models and data successfully loaded in {end_time - start_time:.2f} seconds.")

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup logic: loading resources."""
    logging.info("🚀 Application startup initiated...")
    try:
        load_resources()
        logging.info("✅ Application ready to accept requests.")
    except Exception as e:
        logging.critical(f"❌ CRITICAL ERROR during startup: {e}", exc_info=True)
        raise RuntimeError("Application startup failed") from e
    yield
    logging.info("🛑 Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Question Answering API (Embedder Only)",
    version="1.2.0", # Versi baru
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

# --- Fungsi Caching Embedding Pertanyaan ---
@functools.lru_cache(maxsize=256)
def get_question_embedding(question: str) -> np.ndarray:
    if embedder is None:
        logging.error("❌ Embedder not loaded when trying to encode question!")
        raise RuntimeError("Embedder not available")
    return embedder.encode([question], convert_to_numpy=True)[0] # Ambil embedding pertama

# --- Proses Pertanyaan (Embedder + Cosine Similarity)
def answer_question(question: str, top_n: int = 3) -> str:
    """
    Processes question using Sentence Transformer and semantic search.
    Relies on globally loaded embedder, all_chunks, chunk_embeddings.
    """

    if embedder is None or chunk_embeddings is None or all_chunks is None:
        logging.critical("❌ CRITICAL: Core resources not loaded! Check startup logs.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Resources not loaded.")

    logging.info(f"🔍 Processing question: {question[:100]}...")
    start_process_time = time.time()

    try:
        question_embedding = get_question_embedding(question)
        question_embedding_reshaped = question_embedding.reshape(1, -1)

    except RuntimeError as e:
         logging.error(f"Failed to get question embedding: {e}")
         raise HTTPException(status_code=503, detail="Service temporarily unavailable: Embedder failed.")
    except Exception as e:
         logging.error(f"Unexpected error getting question embedding: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal error during question processing.")

    try:
        logging.debug(f"   Searching top {top_n} chunks using semantic search...")
        hits = util.semantic_search(question_embedding_reshaped, chunk_embeddings, top_k=top_n)
        hits = hits[0]

    except Exception as e:
        logging.error(f"Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during search.")

    # 3. Ambil Teks Chunk yang Relevan
    relevant_chunks = []
    logging.debug("   Relevant chunks found:")
    for hit in hits:
        chunk_index = hit['corpus_id']
        similarity_score = hit['score']
        if 0 <= chunk_index < len(all_chunks):
            logging.debug(f"     - Chunk Index: {chunk_index}, Score: {similarity_score:.4f}")
            relevant_chunks.append(all_chunks[chunk_index])
        else:
            logging.warning(f"     - Invalid chunk index {chunk_index} found in search results (max index: {len(all_chunks)-1}).")

    if not relevant_chunks:
        logging.warning("   -> No relevant chunks found after search.")
        return "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda saat ini."

    # 4. Gabungkan Jawaban (sebelum post-processing)
    raw_answer = "\n\n".join(relevant_chunks)
    # logging.debug(f"   Raw answer constructed: {raw_answer[:200]}...") # Uncomment for debugging

    end_process_time = time.time()
    logging.info(f"✅ Question processed in {end_process_time - start_process_time:.2f} seconds.")

    return raw_answer

# --- Fungsi Post-Processing Jawaban (dari Colab) ---
def post_process_answer(answer: str) -> str:
    logging.debug("   Post-processing answer...")
    # Ganti pemisah chunk asli (\n\n) dengan spasi
    text_for_tokenize = answer.replace("\n\n", " ")
    try:
        sentences = sent_tokenize(text_for_tokenize)
    except Exception as e:
        logging.error(f"Error during sentence tokenization: {e}. Returning raw answer.", exc_info=True)
        return answer # Fallback

    unique_sentences = list(dict.fromkeys(sentences))
    bulleted_list = []
    min_sentence_length = 15
    for sentence in unique_sentences:
        cleaned_sentence = sentence.strip()
        if len(cleaned_sentence) > min_sentence_length and re.search(r'[a-zA-Z]', cleaned_sentence):
             formatted_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
             bulleted_list.append(f"* {formatted_sentence}")

    if not bulleted_list:
         logging.warning("   -> No valid sentences remained after post-processing.")
         return answer if answer.strip() else "Tidak ada informasi detail yang dapat ditampilkan."

    final_output = "\n".join(bulleted_list)
    logging.debug(f"   -> Post-processing complete. Generated {len(bulleted_list)} bullets.")
    return final_output

# --- API Endpoints ---
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", response_model=Dict[str, str])
async def ask_chatbot(request: QuestionRequest):
    """Endpoint to receive a question and return a processed answer."""
    question = request.question
    logging.info(f"📩 Received API request for question: {question[:100]}...")

    if not question or not question.strip():
        logging.warning("⚠️ Received empty question.")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        raw_answer = answer_question(question, top_n=3)
        processed_answer = post_process_answer(raw_answer)
        return {"answer": processed_answer}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.exception("❌ An unexpected error occurred processing the /ask request:")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/logs", response_model=Dict[str, List[str]])
async def get_logs():
    """Returns the last 50 lines of the application log file."""
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
    """Basic health check, indicates if core resources are loaded."""
    status = "ok"
    issues = []
    if embedder is None: issues.append("Embedder not loaded")
    if chunk_embeddings is None: issues.append("Embeddings not loaded")
    if all_chunks is None: issues.append("Chunks not loaded")

    if issues:
        status = f"degraded ({'; '.join(issues)})"
        logging.warning(f"⚠️ Health check status: {status}")

    return {"status": status, "device": device}

@app.get("/", response_model=Dict[str, str])
def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "DocuQuery is running!"}