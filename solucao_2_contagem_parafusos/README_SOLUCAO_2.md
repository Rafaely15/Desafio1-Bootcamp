# Solucao 2 - Sistema Web de Contagem de Parafusos

Esta pasta contem a segunda solucao do Desafio 1: uma aplicacao web em FastAPI para contagem automatica de parafusos com YOLOv11.

## Como executar

1. Instale as dependencias:

```powershell
pip install -r requirements.txt
```

2. Inicie o sistema:

```powershell
.\start_web.ps1
```

ou:

```powershell
.\start_web.bat
```

3. Acesse no navegador:

```text
http://localhost:8000
```

## Modelo

O modelo utilizado pela aplicacao esta em:

```text
runs_screws/yolo11_screws-2/weights/best.pt
```

## Relatorio tecnico

O PDF pronto para avaliacao esta em:

```text
docs/pdf/relatorio_tecnico.pdf
```

O fonte LaTeX e as imagens usadas no relatorio estao em:

```text
docs/latex/
```

Para recompilar o relatorio, execute dentro de `docs/latex`:

```powershell
pdflatex relatorio_tecnico.tex
pdflatex relatorio_tecnico.tex
```

## Observacoes

- O banco `contagens.db` nao foi incluido para a entrega comecar sem registros anteriores.
- As pastas de uploads e resultados sao criadas automaticamente pela aplicacao.
- O dashboard geral mantem historico e separa os registros por dia.
- O fechamento do dia gera CSV, PDF e JSON em `outputs/fechamentos/`.
