@echo off
call myvenv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause