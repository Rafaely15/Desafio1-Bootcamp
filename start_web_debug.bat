@echo off
cd /d "%~dp0"
echo Iniciando em modo debug...
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level debug
echo.
echo O servidor foi encerrado. Se apareceu erro acima, copie a mensagem para diagnostico.
pause
