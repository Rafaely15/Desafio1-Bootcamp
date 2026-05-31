@echo off
echo ============================================================
echo  Metal Mecanica — Contador de Parafusos (Visao Classica v5)
echo ============================================================
echo.

pip install -r requirements.txt --quiet

echo.
echo Iniciando servidor...
echo.
echo  Computador : http://127.0.0.1:8001
echo  Celular    : veja o IP exibido na pagina inicial
echo.

uvicorn main:app --host 0.0.0.0 --port 8001
pause
