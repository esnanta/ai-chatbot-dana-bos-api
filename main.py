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
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        ALL_CHUNKS = json.load(f)

    # Inisialisasi model
    EMBEDDER = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6")

    # Load embeddings dan FAISS index jika tersedia
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(FAISS_INDEX_FILE):
        CHUNK_EMBEDDINGS = np.load(EMBEDDING_FILE)
        INDEX = faiss.read_index(FAISS_INDEX_FILE)
        logging.info("Loaded embeddings and FAISS index from file.")
    else:
        INDEX = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        CHUNK_EMBEDDINGS = EMBEDDER.encode(ALL_CHUNKS, convert_to_numpy=True)
        INDEX.add(CHUNK_EMBEDDINGS)
        np.save(EMBEDDING_FILE, CHUNK_EMBEDDINGS)
        faiss.write_index(INDEX, FAISS_INDEX_FILE)
        logging.info("Created and saved embeddings and FAISS index.")

    end_time = time.time()
    logging.info(f"Models and data loaded in {end_time - start_time:.2f} seconds.")

except Exception as e:
    logging.exception("Error during eager loading:")
    raise

# Fungsi Utilitas
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

def answer_question(question: str, chunks: list[str], index: faiss.Index, embedder: SentenceTransformer, cross_encoder_model: CrossEncoder, top_n: int = 3) -> str:
    try:
        filtered_chunks = filter_chunks_by_keywords(question, chunks)
        if not filtered_chunks:
            return "Maaf, saya tidak dapat menemukan informasi yang sesuai."

        question_embedding = embedder.encode([question], convert_to_numpy=True)
        D, I = index.search(question_embedding, min(top_n * 2, len(CHUNK_EMBEDDINGS)))
        candidates = [chunks[i] for i in I[0]]

        pairs = [(question, chunk) for chunk in candidates]
        scores = cross_encoder_model.predict(pairs, batch_size=4)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return "\n".join([candidates[i] for i in top_indices])
    except Exception as e:
        logging.exception(f"Error in answer_question: {e}")
        return "Error: Terjadi kesalahan dalam memproses pertanyaan."

# Model Request
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_chatbot(request: QuestionRequest):
    try:
        logging.info(f"Received question: {request.question}")
        if not request.question:
            raise HTTPException(status_code=400, detail="No question provided")
        raw_answer = answer_question(request.question, ALL_CHUNKS, INDEX, EMBEDDER, CROSS_ENCODER_MODEL, top_n=3)
        return {"answer": post_process_answer(raw_answer)}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("Error processing request:")
        raise HTTPException(status_code=500, detail="An error occurred processing the request")

@app.get("/")
def read_root():
    return {"message": "API is running!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
