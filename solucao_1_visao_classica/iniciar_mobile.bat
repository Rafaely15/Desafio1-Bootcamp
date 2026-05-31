@echo off
echo ============================================================
echo  Metal Mecanica — Acesso via celular (tunel publico)
echo ============================================================
echo.

pip install -r requirements.txt --quiet

echo.
echo Iniciando servidor na porta 8001...
start "Servidor FastAPI" cmd /k "uvicorn main:app --host 0.0.0.0 --port 8001"

timeout /t 3 /nobreak >nul

echo.
echo Abrindo tunel publico (Cloudflare)...
echo Aguarde o link aparecer na proxima janela — copie e acesse no celular.
echo (Nao precisa estar na mesma rede Wi-Fi)
echo.
start "Tunel Cloudflare" cmd /k "cloudflared.exe tunnel --url http://localhost:8001"

echo.
echo Acesso local (mesma rede Wi-Fi): http://127.0.0.1:8001
echo Acesso publico: veja o link na janela "Tunel Cloudflare"
echo.
pause
