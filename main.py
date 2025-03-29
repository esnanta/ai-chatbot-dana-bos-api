import os
import json
import faiss
import numpy as np
import uvicorn
import nltk
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel
import time

nltk.download('punkt')
nltk.download('punkt_tab')

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # ✅ Ubah ke DEBUG untuk debugging lebih detail
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # ✅ Logging ke console (berguna untuk Render)
        logging.FileHandler("app.log", mode="a")  # ✅ Simpan ke file untuk debugging lebih lanjut
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
EMBEDDING_DIMENSION = 384  # Dimensi embedding model MiniLM
EMBEDDING_FILE = os.path.join(BASE_DIR, "chunk_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")
NPROBE = 5  # Jumlah cluster yang dicari saat query

# ===============================
# 2. MUAT DATA & MODEL
# EAGER LOADING
# ===============================
logging.info("Loading models and data...")
start_time = time.time()

try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        ALL_CHUNKS = json.load(f)
        logging.info(f"✅ Loaded {len(ALL_CHUNKS)} chunks.")

    # Inisialisasi model
    EMBEDDER = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    logging.info("✅ SentenceTransformer model loaded.")

    CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6")
    logging.info("✅ CrossEncoder model loaded.")

    # Load FAISS Index
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
# 3. FUNGSI UTILITAS
# ===============================
def extract_keywords(question, top_n=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return set(feature_array[tfidf_sorting][:top_n])

def filter_chunks_by_keywords(question, chunks):
    keywords = extract_keywords(question)
    return [chunk for chunk in chunks if any(keyword.lower() in chunk.lower() for keyword in keywords)] or chunks

def post_process_answer(answer):
    sentences = sent_tokenize(answer)
    unique_sentences = list(dict.fromkeys(sentences))
    return "<br><br>".join([f"* {sentence}" for sentence in unique_sentences if len(sentence.strip()) > 10])


def answer_question(question: str, chunks: list[str], index: faiss.Index, embedder: SentenceTransformer,
                    cross_encoder_model: CrossEncoder, top_n: int = 3) -> str:
    try:
        logging.info(f"🔍 Processing question: {question}")

        # Encode pertanyaan
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        logging.debug(f"✅ Question embedding generated: {question_embedding.shape}")

        # Cari top_n * 2 kandidat di FAISS
        D, I = index.search(question_embedding, min(top_n * 2, len(CHUNK_EMBEDDINGS)))
        candidates = [chunks[i] for i in I[0]]
        logging.debug(f"🔍 FAISS retrieved {len(candidates)} candidates.")

        # Re-rank menggunakan CrossEncoder
        pairs = [(question, chunk) for chunk in candidates]
        scores = cross_encoder_model.predict(pairs, batch_size=4)
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
            logging.warning("⚠️ No question provided in request.")
            raise HTTPException(status_code=400, detail="No question provided")

        raw_answer = answer_question(request.question, ALL_CHUNKS, INDEX_FAISS, EMBEDDER, CROSS_ENCODER_MODEL, top_n=3)
        processed_answer = raw_answer.replace("\n", "<br><br>")

        logging.info(f"📤 Sending API response: {processed_answer}")
        return {"answer": processed_answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("❌ Error processing request:")
        raise HTTPException(status_code=500, detail="An error occurred processing the request")

@app.get("/")
def read_root():
    logging.info("✅ Root endpoint accessed.")
    return {"message": "API is running!"}

# ===============================
# 5. MENJALANKAN SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logging.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)