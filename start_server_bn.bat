@echo off
cd /d %~dp0
call .venv\Scripts\activate
start python run_server.py --llm-base-url http://localhost:8000/v1 --best-of-n-enabled
python run_gradio.py
pause
