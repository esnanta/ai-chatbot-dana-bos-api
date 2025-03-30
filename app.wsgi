from main import app  # Pastikan "main" adalah nama file utama yang berisi FastAPI app

# Gunicorn akan menggunakan objek 'app' ini
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)