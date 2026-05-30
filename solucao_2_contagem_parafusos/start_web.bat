@echo off
cd /d "%~dp0"
echo Iniciando Sistema de Contagem de Parafusos...
echo.
echo Acesse no computador:
echo   http://localhost:8000
echo.
echo Para parar o servidor, pressione CTRL+C nesta janela.
echo.
set "PYTHON=%USERPROFILE%\anaconda3\envs\yolov11\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo O servidor foi encerrado. Se apareceu erro acima, copie a mensagem para diagnostico.
pause
