import os
import json
import faiss
import numpy as np
import uvicorn
import nltk

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel
import time
import logging
import torch  # Import torch untuk pengecekan CUDA

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Konstanta dan Inisialisasi Global
BASE_DIR = "knowledge_base"
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
EMBEDDING_DIMENSION = 384  # Dimensi embedding model MiniLM
EMBEDDING_FILE = os.path.join(BASE_DIR, "chunk_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")

# Muat data dan model saat aplikasi dimulai (Eager Loading)
logging.info("Loading models and data...")
start_time = time.time()

try:
    # Muat chunks dari file
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        ALL_CHUNKS = json.load(f)

    # Inisialisasi model SentenceTransformer dan CrossEncoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Tentukan device
    logging.info(f"Using device: {device}")

    EMBEDDER = SentenceTransformer("paraphrase-MiniLM-L3-v2").to(device)
    CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").to(device)


    # Load embeddings dan FAISS index dari file jika ada, jika tidak, buat baru
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
        CHUNK_EMBEDDINGS = np.load(EMBEDDING_FILE)
        INDEX = faiss.read_index(FAISS_INDEX_FILE)
        logging.info(f"Loaded embeddings and FAISS index from file.")
    else:
        # Inisialisasi dan latih index FAISS
        INDEX = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        CHUNK_EMBEDDINGS = EMBEDDER.encode(ALL_CHUNKS, convert_to_numpy=True)
        INDEX.add(CHUNK_EMBEDDINGS)

        # Simpan embeddings dan FAISS index ke file
        np.save(EMBEDDING_FILE, CHUNK_EMBEDDINGS)
        faiss.write_index(INDEX, FAISS_INDEX_FILE)
        logging.info(f"Created and saved embeddings and FAISS index.")

    end_time = time.time()
    logging.info(f"Models and data loaded in {end_time - start_time:.2f} seconds.")

except FileNotFoundError:
    logging.error(f"Chunks file not found at {CHUNKS_FILE}")
    ALL_CHUNKS = None
    INDEX = None
    EMBEDDER = None
    CROSS_ENCODER_MODEL = None
    raise  # Re-raise exception agar aplikasi berhenti saat startup
except Exception as e:
    logging.exception("Error during eager loading:")  # Log detail error
    ALL_CHUNKS = None
    INDEX = None
    EMBEDDER = None
    CROSS_ENCODER_MODEL = None
    raise  # Re-raise exception agar aplikasi berhenti saat startup

# Fungsi Utilitas
def extract_keywords(question, top_n=5):
    """Ekstraksi kata kunci menggunakan TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return set(top_keywords)

def filter_chunks_by_keywords(question, chunks):
    """Filter chunks berdasarkan kata kunci."""
    keywords = extract_keywords(question)
    filtered_chunks = [chunk for chunk in chunks if any(keyword.lower() in chunk.lower() for keyword in keywords)]
    return filtered_chunks if filtered_chunks else chunks

def post_process_answer(answer):
    """Memproses jawaban untuk menghasilkan daftar bullet yang terpisah."""
    sentences = sent_tokenize(answer)

    # Hapus duplikasi dan urutkan kalimat agar lebih jelas
    unique_sentences = list(dict.fromkeys(sentences))

    # Filter kalimat yang panjangnya lebih dari 10 karakter
    filtered_sentences = [sentence.strip() for sentence in unique_sentences if len(sentence.strip()) > 10]

    # Format sebagai bullet list dengan menambahkan karakter newline setelah setiap item
    bulleted_list = "<br><br>".join([f"* {sentence}" for sentence in filtered_sentences])

    return bulleted_list

# Fungsi Utama: Menjawab Pertanyaan
def answer_question(question: str, chunks: list[str], index: faiss.Index, embedder: SentenceTransformer, cross_encoder_model: CrossEncoder, top_n: int = 3) -> str:
    """Menjawab pertanyaan berdasarkan knowledge base."""
    try:
        if not ALL_CHUNKS or not INDEX or not EMBEDDER or not CROSS_ENCODER_MODEL:
            raise RuntimeError("Models or data not loaded properly.")

        # Filter chunk berdasarkan kata kunci pertanyaan
        filtered_chunks = filter_chunks_by_keywords(question, chunks)

        if not filtered_chunks:
            return "Maaf, saya tidak dapat menemukan informasi yang sesuai."

        # Lakukan embedding hanya pada pertanyaan (bukan ulang chunk)
        question_embedding = embedder.encode([question], convert_to_numpy=True)

        # Cari similarity dengan FAISS yang sudah dimuat dari file
        D, I = index.search(question_embedding, min(top_n * 2, len(CHUNK_EMBEDDINGS)))

        # Ambil kandidat chunk berdasarkan FAISS
        candidates = [chunks[i] for i in I[0]]

        # Gunakan Cross-Encoder untuk memilih chunk terbaik
        pairs = [(question, chunk) for chunk in candidates]
        scores = cross_encoder_model.predict(pairs)
        top_indices = np.argsort(scores)[::-1][:top_n]

        context = "\n".join([candidates[i] for i in top_indices])
        return context
    except Exception as e:
        logging.exception(f"Error in answer_question: {e}")
        return "Error: Terjadi kesalahan dalam memproses pertanyaan."

# Model Request (Pydantic)
class QuestionRequest(BaseModel):
    question: str

# API Endpoints
@app.post("/ask")
async def ask_chatbot(request: QuestionRequest):
    """Endpoint untuk menerima pertanyaan dan memberikan jawaban."""
    try:
        logging.info(f"Received question: {request.question}")
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")

        raw_answer = answer_question(
            request.question,
            ALL_CHUNKS,
            INDEX,
            EMBEDDER,
            CROSS_ENCODER_MODEL,
            top_n=3
        )
        processed_answer = post_process_answer(raw_answer)
        return {"answer": processed_answer}

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("Error processing request:")
        raise HTTPException(status_code=500, detail="An error occurred processing the request")

@app.get("/")
def read_root():
    """Endpoint untuk memeriksa apakah API berjalan."""
    return {"message": "API is running!"}

# Main Execution
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)