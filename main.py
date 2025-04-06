import os
import json
import numpy as np
import nltk
import logging
import functools # Untuk lru_cache
import torch
import time
import re # Untuk post_process_answer

from typing import List, Dict, Optional, Tuple # Tambahkan Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Import SentenceTransformer, CrossEncoder, dan util
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk.tokenize import sent_tokenize # Untuk post_process_answer
from pydantic import BaseModel

from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError

logging.info("Attempting explicit Hugging Face login...")
print("Attempting explicit Hugging Face login...")
hf_token = os.environ.get("HF_TOKEN_READ")
if hf_token:
    logging.info("HF_TOKEN environment variable found.")
    print("HF_TOKEN environment variable found.")
    try:
        # Coba login menggunakan token dari environment variable
        login(token=hf_token)
        logging.info("✅ Explicit Hugging Face login successful.")
        print("✅ Explicit Hugging Face login successful.")
    except HfHubHTTPError as e: # Tangkap error login spesifik
        logging.error(f"!!! Explicit Hugging Face login failed: {e} !!!")
        print(f"!!! PRINT ERROR Explicit Hugging Face login failed: {e} !!!")
        # Anda bisa memutuskan apakah akan melanjutkan atau menghentikan di sini
        # raise RuntimeError("Failed to login to Hugging Face Hub") from e
        logging.warning("Proceeding without guaranteed authenticated session.")
    except Exception as e:
        logging.exception("!!! Unexpected error during explicit Hugging Face login !!!")
        print(f"!!! PRINT ERROR Unexpected error during explicit Hugging Face login: {e} !!!")
else:
    logging.warning("HF_TOKEN environment variable not found. Proceeding with anonymous access.")
    print("HF_TOKEN environment variable not found. Proceeding with anonymous access.")


# --- Konstanta ---
RENDER_DATA_DIR = os.environ.get("RENDER_DATA_DIR", "/var/data") # Gunakan env var jika diset
NLTK_DATA_PATH = os.path.join(RENDER_DATA_DIR, "nltk_data")
MODEL_CACHE_PATH = os.path.join(RENDER_DATA_DIR, "model_cache")
LOG_FILE_PATH = os.path.join(RENDER_DATA_DIR, "logs", "app.log")

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunks.json")
EMBEDDING_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "chunk_embeddings.npy")

# --- Parameter Tambahan ---
# Berapa banyak kandidat yang diambil oleh bi-encoder untuk di-rerank oleh cross-encoder
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", 10))
# Berapa banyak hasil akhir yang dikembalikan setelah re-ranking
DEFAULT_FINAL_TOP_N = 3

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
ensure_nltk_data('punkt_tab') # Untuk sent_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Using device: {device}")

# --- Global Variables ---
all_chunks: Optional[List[str]] = None
embedder: Optional[SentenceTransformer] = None # Bi-Encoder (Retriever)
cross_encoder: Optional[CrossEncoder] = None   # Cross-Encoder (Re-ranker)
chunk_embeddings: Optional[np.ndarray] = None

def load_resources():

    global all_chunks, embedder, cross_encoder, chunk_embeddings
    logging.info("🔄 Loading models and data (Bi-Encoder + Cross-Encoder)...")
    start_time = time.time()

    # --- Pilih Model Bi-Encoder (Embedder) ---
    # Pastikan ini SAMA dengan model yang digunakan untuk membuat EMBEDDING_FILE
    # selected_embedder_model = "paraphrase-MiniLM-L3-v2"
    selected_embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
    # selected_embedder_model = "multi-qa-MiniLM-L6-cos-v1"
    logging.info(f"Selected Bi-Encoder (Embedder) model: {selected_embedder_model}")
    logging.warning(f"🚨 ENSURE '{EMBEDDING_FILE}' was generated using '{selected_embedder_model}'!")

    # --- Pilih Model Cross-Encoder (Re-ranker) ---
    # Model Cross-Encoder yang populer untuk re-ranking
    selected_cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logging.info(f"Selected Cross-Encoder (Re-ranker) model: {selected_cross_encoder_model}")

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

    # Load Sentence Transformer (Bi-Encoder)
    try:
        logging.info(f"💾 Loading Bi-Encoder model: '{selected_embedder_model}' (cache: {MODEL_CACHE_PATH})...")
        embedder = SentenceTransformer(
            selected_embedder_model,
            cache_folder=MODEL_CACHE_PATH,
            device=device
        )
        logging.info("✅ Bi-Encoder model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Bi-Encoder model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Bi-Encoder: {selected_embedder_model}") from e

    # Load Cross-Encoder
    try:
        logging.info(f"💾 Loading Cross-Encoder model: '{selected_cross_encoder_model}' (cache: {MODEL_CACHE_PATH})...")
        cross_encoder = CrossEncoder(
            selected_cross_encoder_model,
            max_length=512,
            cache_folder=MODEL_CACHE_PATH,
            device=device
        )
        logging.info("✅ Cross-Encoder model loaded.")
    except Exception as e:
        logging.error(f"❌ Error loading Cross-Encoder model: {e}", exc_info=True)
        # Anda bisa memilih untuk melanjutkan tanpa cross-encoder atau gagal total
        # raise RuntimeError(f"Failed to load Cross-Encoder: {selected_cross_encoder_model}") from e
        logging.warning("⚠️ Cross-Encoder loading failed. Proceeding without re-ranking.")
        cross_encoder = None # Set ke None jika gagal

    # Dapatkan dimensi embedding yang diharapkan dari Bi-Encoder
    expected_dim = None
    if embedder:
        try:
            expected_dim = embedder.get_sentence_embedding_dimension()
            logging.info(f"Bi-Encoder embedding dimension: {expected_dim}")
        except Exception as e:
            logging.error(f"Could not get embedding dimension from Bi-Encoder model: {e}")

    # Load Embeddings
    try:
        if not os.path.exists(EMBEDDING_FILE):
             logging.error(f"❌ Embedding file not found at {EMBEDDING_FILE}")
             raise FileNotFoundError(f"Embedding file not found at {EMBEDDING_FILE}. Generate it first matching model '{selected_embedder_model}'.")
        logging.info(f"💾 Loading embeddings from {EMBEDDING_FILE}...")
        chunk_embeddings = np.load(EMBEDDING_FILE)
        logging.info(f"✅ Loaded embeddings with shape {chunk_embeddings.shape}.")

        # --- Validasi Krusial ---
        if len(all_chunks) != chunk_embeddings.shape[0]:
             logging.error(f"❌ Mismatch! Chunks count ({len(all_chunks)}) != Embeddings count ({chunk_embeddings.shape[0]}). Check {CHUNKS_FILE} and {EMBEDDING_FILE}.")
             raise ValueError("Mismatch between number of chunks and embeddings.")
        if expected_dim is not None and chunk_embeddings.shape[1] != expected_dim:
            logging.error(f"❌ Embedding dimension mismatch! Expected {expected_dim} (from model '{selected_embedder_model}'), but file has {chunk_embeddings.shape[1]}. Regenerate '{EMBEDDING_FILE}'.")
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
        # Berhenti jika gagal memuat resource krusial (embedder, chunks, embeddings)
        if embedder is None or all_chunks is None or chunk_embeddings is None:
             raise RuntimeError("Application startup failed due to critical resource loading error") from e
        # Jika hanya cross-encoder yang gagal, mungkin bisa lanjut dengan warning
        elif cross_encoder is None:
             logging.warning("⚠️ Proceeding without Cross-Encoder due to loading failure.")
        else:
             raise RuntimeError("Application startup failed") from e

    yield
    logging.info("🛑 Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Question Answering API (Bi-Encoder + Cross-Encoder)",
    version="2.0.0", # Versi baru dengan Cross-Encoder
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
    logging.debug(f"   Encoding question (Bi-Encoder): {question[:50]}...")
    return embedder.encode([question], convert_to_numpy=True)[0] # Ambil embedding pertama

# --- Proses Pertanyaan (Bi-Encoder Retrieval + Cross-Encoder Re-ranking) ---
def answer_question(question: str, retriever_top_k: int = RETRIEVER_TOP_K, final_top_n: int = DEFAULT_FINAL_TOP_N) -> str:
    """
    Processes question using Bi-Encoder for retrieval and Cross-Encoder for re-ranking.
    """
    if embedder is None or chunk_embeddings is None or all_chunks is None:
        logging.critical("❌ CRITICAL: Core resources (Embedder, Embeddings, Chunks) not loaded! Check startup logs.")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable: Core resources not loaded.")

    logging.info(f"🔍 Processing question: {question[:100]}...")
    start_process_time = time.time()

    # 1. Retrieval dengan Bi-Encoder
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
        logging.info(f"   Performing semantic search (Bi-Encoder) for top {retriever_top_k} candidates...")
        start_retrieval_time = time.time()
        # Ambil lebih banyak kandidat untuk di-re-rank
        hits = util.semantic_search(question_embedding_reshaped, chunk_embeddings, top_k=retriever_top_k)
        hits = hits[0] # Hasil untuk query pertama
        retrieval_time = time.time() - start_retrieval_time
        logging.info(f"   Semantic search completed in {retrieval_time:.3f} seconds. Found {len(hits)} candidates.")

    except Exception as e:
        logging.error(f"Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during search.")

    # Ambil teks chunk kandidat
    retrieved_chunks_info: List[Dict[str, any]] = []
    for hit in hits:
        chunk_index = hit['corpus_id']
        if 0 <= chunk_index < len(all_chunks):
            retrieved_chunks_info.append({
                "index": chunk_index,
                "text": all_chunks[chunk_index],
                "retriever_score": hit['score']
            })
        else:
            logging.warning(f"     - Invalid chunk index {chunk_index} found in search results (max index: {len(all_chunks)-1}). Skipping.")

    if not retrieved_chunks_info:
        logging.warning("   -> No relevant chunks found after initial retrieval.")
        return "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda saat ini."

    # 2. Re-ranking dengan Cross-Encoder (jika tersedia)
    if cross_encoder:
        logging.info(f"   Re-ranking top {len(retrieved_chunks_info)} candidates using Cross-Encoder...")
        start_rerank_time = time.time()

        # Siapkan pasangan [pertanyaan, chunk_teks]
        cross_inp = [[question, chunk_info["text"]] for chunk_info in retrieved_chunks_info]

        try:
            # Hitung skor dengan Cross-Encoder
            cross_scores = cross_encoder.predict(cross_inp)
            rerank_time = time.time() - start_rerank_time
            logging.info(f"   Cross-Encoder prediction completed in {rerank_time:.3f} seconds.")

            # Tambahkan skor cross-encoder ke info chunk
            for idx, score in enumerate(cross_scores):
                retrieved_chunks_info[idx]["cross_score"] = float(score) # Konversi ke float jika perlu

            # Urutkan ulang berdasarkan skor cross-encoder (dari tertinggi ke terendah)
            reranked_chunks_info = sorted(retrieved_chunks_info, key=lambda x: x.get("cross_score", -float('inf')), reverse=True)

            # Pilih top-N final dari hasil re-ranking
            final_chunks_info = reranked_chunks_info[:final_top_n]
            logging.debug("   Top chunks after re-ranking:")
            for i, chunk_info in enumerate(final_chunks_info):
                 logging.debug(f"     {i+1}. Index: {chunk_info['index']}, Retriever Score: {chunk_info['retriever_score']:.4f}, Cross Score: {chunk_info.get('cross_score', 'N/A'):.4f}")

        except Exception as e:
            logging.error(f"Error during Cross-Encoder re-ranking: {e}", exc_info=True)
            # Fallback: Gunakan hasil retriever jika re-ranking gagal
            logging.warning("   ⚠️ Falling back to Bi-Encoder results due to re-ranking error.")
            # Urutkan berdasarkan skor retriever asli
            retrieved_chunks_info.sort(key=lambda x: x['retriever_score'], reverse=True)
            final_chunks_info = retrieved_chunks_info[:final_top_n]

    else:
        # Jika tidak ada Cross-Encoder, langsung gunakan hasil retriever
        logging.warning("   Skipping re-ranking because Cross-Encoder is not available.")
        # Urutkan berdasarkan skor retriever (seharusnya sudah terurut, tapi untuk keamanan)
        retrieved_chunks_info.sort(key=lambda x: x['retriever_score'], reverse=True)
        final_chunks_info = retrieved_chunks_info[:final_top_n]
        logging.debug("   Top chunks from Bi-Encoder (no re-ranking):")
        for i, chunk_info in enumerate(final_chunks_info):
             logging.debug(f"     {i+1}. Index: {chunk_info['index']}, Retriever Score: {chunk_info['retriever_score']:.4f}")


    # 3. Gabungkan Jawaban Final
    final_answer_parts = []
    separator = "\n\n====================\n\n"
    logging.debug("   Formatting final answer with separators:")
    for i, chunk_info in enumerate(final_chunks_info):
        chunk_text = chunk_info["text"].strip()
        header = f"[Sumber Konteks {i+1}]"
        formatted_part = f"{header}\n{chunk_text}" # Gabungkan header dan teks chunk
        final_answer_parts.append(formatted_part)
        logging.debug(f"     Part {i+1} added (Length: {len(chunk_text)} chars)")

    # Gabungkan semua bagian yang sudah diformat menggunakan separator
    raw_answer_with_separators = separator.join(final_answer_parts)

    end_process_time = time.time()
    logging.info(f"✅ Question processed (Retrieval + Re-ranking/Selection) in {end_process_time - start_process_time:.2f} seconds.")

    return raw_answer_with_separators

# --- Fungsi Post-Processing Jawaban (Sama seperti sebelumnya) ---
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
         # Kembalikan jawaban mentah jika post-processing mengosongkannya
         return answer if answer.strip() else "Tidak ada informasi detail yang dapat ditampilkan."


    final_output = "\n".join(bulleted_list)
    logging.debug(f"   -> Post-processing complete. Generated {len(bulleted_list)} bullets.")
    return final_output

# --- API Endpoints ---
class QuestionRequest(BaseModel):
    question: str
    top_n: Optional[int] = DEFAULT_FINAL_TOP_N # Berapa hasil akhir yang diinginkan

@app.post("/ask", response_model=Dict[str, str])
async def ask_chatbot(request: QuestionRequest):
    """
    Endpoint untuk menerima pertanyaan, melakukan retrieval & re-ranking,
    dan mengembalikan jawaban yang diproses.
    """
    question = request.question
    final_top_n = request.top_n if request.top_n > 0 else DEFAULT_FINAL_TOP_N
    logging.info(f"📩 Received API request for question: {question[:100]}... (Returning top {final_top_n})")

    if not question or not question.strip():
        logging.warning("⚠️ Received empty question.")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Panggil answer_question dengan retriever_top_k (konstan) dan final_top_n (dari request)
        raw_answer = answer_question(
            question,
            retriever_top_k=RETRIEVER_TOP_K,
            final_top_n=final_top_n
        )
        processed_answer = post_process_answer(raw_answer)
        return {"answer": processed_answer}

    except HTTPException as http_exc:
        # Re-raise HTTPException yang sudah ditangani di answer_question
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
    if embedder is None: issues.append("Bi-Encoder (Embedder) not loaded")
    if chunk_embeddings is None: issues.append("Embeddings not loaded")
    if all_chunks is None: issues.append("Chunks not loaded")
    if cross_encoder is None: issues.append("Cross-Encoder (Re-ranker) not loaded/failed") # Tambahkan cek cross-encoder

    if issues:
        # Jika resource krusial hilang, statusnya error
        if "Bi-Encoder (Embedder) not loaded" in issues or \
           "Embeddings not loaded" in issues or \
           "Chunks not loaded" in issues:
            status = f"error ({'; '.join(issues)})"
        # Jika hanya cross-encoder yang hilang, status degraded
        elif "Cross-Encoder (Re-ranker) not loaded/failed" in issues:
            status = f"degraded (Running without re-ranking: {'; '.join(issues)})"
        else:
             status = f"degraded ({'; '.join(issues)})" # Kasus lain jika ada

        logging.warning(f"⚠️ Health check status: {status}")

    return {"status": status, "device": device}


@app.get("/", response_model=Dict[str, str])
def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "DocuQuery (with Cross-Encoder) is running!"}

# --- Untuk menjalankan dengan uvicorn (opsional, jika dijalankan langsung) ---
# if __name__ == "__main__":
#     import uvicorn
#     logging.info("Starting Uvicorn server...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)