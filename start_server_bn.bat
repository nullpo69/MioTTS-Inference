@echo off
cd /d %~dp0
call .venv\Scripts\activate
python run_server.py --llm-base-url http://localhost:8000/v1 --best-of-n-enabled
pause
