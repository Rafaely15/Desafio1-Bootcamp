# Desafio 1 — Contagem de Parafusos

Sistema de contagem automática de parafusos para uso industrial em smartphones. O funcionário se identifica, fotografa o lote e recebe a contagem automática em segundos, com histórico rastreável e exportação de relatórios.

O desafio foi resolvido com **duas soluções complementares**:

| | Solução 1 — Visão Clássica | Solução 2 — YOLOv11 |
|---|---|---|
| Tecnologia | OpenCV + Watershed + DBSCAN | YOLOv11 (Ultralytics) |
| Interface | FastAPI (web responsiva) | FastAPI (web responsiva) |
| GPU necessária | Não | Não (recomendada para treino) |
| Modelo treinado | Não | Sim (`best.pt`) |
| Porta padrão | 8001 | 8000 |

---

## Estrutura

```
desafio-1/
├── solucao_1_visao_classica/   # Solução 1 — OpenCV + Watershed
├── solucao_2_yolo/             # Solução 2 — YOLOv11 (nova solução)
├── Entrega-ao-cliente/         # PDFs de estratégia e guia do usuário
└── logs_run/                   # Logs de execução
```

---

## Solução 1 — Visão Clássica (OpenCV)

Pipeline sem modelo treinado: equalização CLAHE → segmentação adaptativa → morfologia → Watershed → DBSCAN. Totalmente explicável e executa sem GPU.

**Como rodar:**

```bash
cd solucao_1_visao_classica
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

Ou clique em `iniciar.bat` no Windows.

Acesse em: `http://localhost:8001`

Documentação completa: [solucao_1_visao_classica/README.md](solucao_1_visao_classica/README.md)

---

## Solução 2 — YOLOv11 (nova solução)

Detecção por deep learning com o modelo YOLOv11 treinado no dataset Screw v4 (Roboflow). Maior robustez a variações de iluminação, ângulo e sobreposição de parafusos.

**Como rodar (modelo já treinado incluso):**

```bash
cd solucao_2_yolo
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse em: `http://localhost:8000`

**Modelo de produção:** `runs_screws/yolo11_screws-2/weights/best.pt`

**Para re-treinar do zero:**

```bash
# 1. Converter dataset para bounding boxes
python scripts/convert_segments_to_boxes.py \
  --input Screw.v4-dataset-screw4.yolov11 \
  --output dataset_detect

# 2. Treinar
python scripts/train_screws_yolo.py \
  --data dataset_detect/data.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 960

# 3. Validar
python scripts/validate_screws_yolo.py \
  --data dataset_detect/data.yaml \
  --model runs_screws/yolo11_screws-2/weights/best.pt
```

Documentação completa: [solucao_2_yolo/README.md](solucao_2_yolo/README.md)

---

## Download

Pasta completa via Google Drive:
[https://drive.google.com/file/d/1M-knoev37IBSK6P4jWJIQSwxHYDUBPFU/view?usp=sharing](https://drive.google.com/file/d/1M-knoev37IBSK6P4jWJIQSwxHYDUBPFU/view?usp=sharing)

Clone via Git:

```bash
git clone https://github.com/Rafaely15/Desafio1-Bootcamp.git
```

---

## Acesso pelo smartphone

Para acessar pelo celular na mesma rede Wi-Fi:

1. Descubra o IP local do computador: `ipconfig` (Windows)
2. Acesse no celular: `http://<SEU_IP>:<PORTA>`

Para acesso remoto (fora da rede local), use Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

---

Bootcamp CDIA — Programa de Residência em IA — Ana Rafaely — 2026
