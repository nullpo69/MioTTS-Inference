@echo off
cd /d %~dp0
call .venv\Scripts\activate

set OLLAMA_HOST=localhost:8000
start "Ollama" ollama serve

timeout /t 5 /nobreak >nul

start "MioTTS Server" python run_server.py --llm-base-url http://localhost:8000/v1 --llm-model hf.co/Aratako/MioTTS-GGUF:MioTTS-1.2B-BF16.gguf --best-of-n-enabled

python run_gradio.py
pause
