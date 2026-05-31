# Contador de Parafusos — Visão Clássica v5

Sistema web de contagem automática de parafusos usando **visão computacional clássica** (OpenCV + Watershed). O funcionário se identifica, fotografa o lote, recebe a contagem automática, confirma ou corrige o resultado e o sistema salva o histórico com rastreabilidade completa.

---

## Estrutura do projeto

```
projeto_parafusos_v5/
│
├── main.py                      # Servidor FastAPI — interface web principal
├── app.py                       # Versão Streamlit (alternativa)
├── screw_counter.py             # Algoritmo central de contagem (OpenCV)
├── contador_parafusos_web.py    # Adaptador do pipeline para uso web
├── evaluation.py                # Avaliação quantitativa do pipeline
├── metrics.py                   # Métricas de desempenho
├── database_utils.py            # Utilitários de banco de dados (Streamlit)
│
├── app/                         # Pacote da interface FastAPI
│   ├── config.py                # Caminhos e configurações
│   ├── database.py              # SQLAlchemy + SQLite
│   ├── models.py                # Modelo Contagem
│   ├── templates/               # HTML (index, result, dashboard)
│   └── static/                  # CSS e assets
│       ├── css/style.css
│       └── assets/capture_guide.png
│
├── data/raw/                    # Imagens de teste (img1–img5)
├── outputs/                     # Resultados de avaliação e relatórios
├── Desafio1_Contagem_Parafusos_v3.ipynb  # Notebook de desenvolvimento
│
├── requirements.txt
├── iniciar.bat                  # Launcher Windows (FastAPI)
├── .gitignore
└── legado/                      # Versões anteriores (Flask, docs antigos)
```

---

## Como instalar

```bash
pip install -r requirements.txt
```

---

## Como rodar — Interface Web (FastAPI)

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Ou simplesmente clique em **`iniciar.bat`** no Windows.

| Acesso | URL |
|---|---|
| Computador | http://127.0.0.1:8001 |
| Celular (mesma rede Wi-Fi) | http://\<IP-local\>:8001 |

O IP local é exibido automaticamente na página inicial do sistema.

---

## Como rodar — Versão Streamlit

```bash
streamlit run app.py
```

Acesse em: http://localhost:8501

---

## Funcionalidades da interface web

| Funcionalidade | Descrição |
|---|---|
| Login por funcionário | Nome, matrícula e setor salvos em cookie (12 h) |
| Captura de foto | Abre câmera direto no celular via `capture="environment"` |
| Contagem automática | Pipeline OpenCV: segmentação adaptativa + watershed |
| Tela de resultado | Imagem processada + contagem + correção manual |
| Dashboard | Histórico agrupado por dia com totais automáticos |
| Fechamento do dia | Gera JSON + CSV + PDF do dia |
| Exportação | CSV geral ou por dia disponível a qualquer momento |

---

## Pipeline de contagem (screw_counter.py)

1. **Pré-processamento** — equalização adaptativa (CLAHE), denoising
2. **Segmentação adaptativa ao ruído** — thresholding local com análise de fundo
3. **Morfologia** — operações de fechamento e abertura para separar objetos
4. **Watershed** — separa parafusos sobrepostos usando Distance Transform + peaks
5. **DBSCAN** — agrupa fragmentos quando necessário
6. **Estimativa final** — razão área/referência + contagem por componente

---

## Avaliação

Execute a avaliação completa nas imagens de teste:

```bash
python evaluation.py
```

Os resultados são salvos em `outputs/`.

---

## Observações

- O pipeline não usa modelo treinado — é totalmente explicável e executa sem GPU.
- As correções manuais salvas pelo app formam uma base para evolução futura com YOLO supervisionado.
- Versões anteriores estão na pasta `legado/` para referência.
