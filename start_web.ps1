Set-Location -Path $PSScriptRoot
Write-Host "Iniciando Sistema de Contagem de Parafusos..."
Write-Host ""
Write-Host "Acesse no computador:"
Write-Host "  http://localhost:8000"
Write-Host ""
Write-Host "Para parar o servidor, pressione CTRL+C nesta janela."
Write-Host ""
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
