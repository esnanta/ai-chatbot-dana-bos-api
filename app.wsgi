from main import app  # Mengimpor aplikasi FastAPI dari main.py

if __name__ != "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)