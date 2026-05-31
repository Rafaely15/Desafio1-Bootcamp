@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo  DESAFIO 1 - SOLUCAO YOLO
echo  Sistema de Contagem de Parafusos
echo ==========================================
echo.
echo Iniciando apenas a solucao YOLO...
echo Pasta atual:
echo   %CD%
echo.

set "PYTHON=%USERPROFILE%\anaconda3\envs\yolov11\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "PORT=8000"

echo Acesse no computador:
echo   http://127.0.0.1:%PORT%
echo.
echo Para acessar no celular, use o IP do computador na mesma rede Wi-Fi:
echo   http://SEU-IP:%PORT%
echo.
echo Para parar o servidor, pressione CTRL+C nesta janela.
echo.

"%PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%

echo.
echo O servidor YOLO foi encerrado.
pause
