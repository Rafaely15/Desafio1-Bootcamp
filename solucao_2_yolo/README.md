# Sistema de Contagem de Parafusos com YOLO11

Projeto para treinar YOLO11/YOLOv11 com Ultralytics, detectar parafusos em fotos, contar as caixas detectadas e disponibilizar uma interface web responsiva para uso por funcionários em smartphone.

## Instalação

```bash
pip install -r requirements.txt
```

Verifique o ambiente:

```bash
python scripts/check_environment.py
```

## Dataset

O dataset pode usar o formato YOLO clássico:

```text
dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
```

Este projeto já inclui um `dataset/data.yaml` apontando para a pasta existente `Screw.v4-dataset-screw4.yolov11`, que está em layout Roboflow:

```text
Screw.v4-dataset-screw4.yolov11/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

Verifique o dataset:

```bash
python scripts/check_dataset.py --dataset Screw.v4-dataset-screw4.yolov11
```

O dataset atual veio do Roboflow com labels em poligonos/segmentacao. Para treinar detecção por bounding boxes, converta uma vez:

```bash
python scripts/convert_segments_to_boxes.py --input Screw.v4-dataset-screw4.yolov11 --output dataset_detect
python scripts/check_dataset.py --dataset dataset_detect
```

Se tiver dados brutos ainda não divididos:

```bash
python scripts/prepare_dataset.py --input raw_dataset --output dataset
```

## Treinamento

Treino recomendado no servidor/PC:

```bash
python scripts/train_screws_yolo.py --data dataset_detect/data.yaml --model yolo11s.pt --epochs 100 --imgsz 960
```

Treino rápido ou pensando em mobile:

```bash
python scripts/train_screws_yolo.py --data dataset/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --name yolo11n_screws
```

`imgsz=640` é mais rápido. Para parafusos pequenos, `imgsz=960` ou `1280` pode melhorar a detecção, usando mais memória.

## Validação

```bash
python scripts/validate_screws_yolo.py --data dataset_detect/data.yaml --model runs_screws/yolo11_screws/weights/best.pt
```

Alta precision indica menos falsos parafusos. Alto recall indica menos parafusos perdidos. Para contagem, os dois precisam ficar equilibrados.

## Notebook de Analise

Para verificar melhor dataset, anotacoes, treino, metricas, inferencias visuais, varredura de confianca e CSV de comparacao, abra:

```text
notebooks/analise_treinamento_parafusos_yolo11.ipynb
```

No VS Code ou Jupyter, execute as celulas em ordem. O notebook usa por padrao `dataset_detect/data.yaml` e `runs_screws/yolo11_screws/weights/best.pt`.

## Contagem no Terminal

Imagem única:

```bash
python scripts/count_screws.py --image exemplos/teste.jpg --model runs_screws/yolo11_screws/weights/best.pt --conf 0.25
```

Pasta inteira:

```bash
python scripts/count_screws.py --source caminho/da/pasta --model runs_screws/yolo11_screws/weights/best.pt --conf 0.25
```

As imagens processadas e o CSV ficam em `outputs/contagens_parafusos.csv`.

## Backend Web

Inicie a aplicação:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

No computador, acesse:

```text
http://localhost:8000
```

No smartphone, conecte o celular na mesma rede Wi-Fi do computador/servidor e acesse:

```text
http://IP_DO_COMPUTADOR:8000
```

Rotas principais:

- `GET /`: envio de foto.
- `POST /predict`: contagem dos parafusos.
- `GET /dashboard`: histórico.
- `GET /export/csv`: exporta todos os registros.
- `GET /export/csv/today`: exporta os registros do dia.
- `GET /health`: status do sistema.

## Banco de Dados

O sistema usa SQLite em `contagens.db`. A tabela `contagens` salva funcionário, data/hora, total, confiança média, imagem original e imagem processada.

No futuro, a troca para PostgreSQL pode ser feita alterando `DATABASE_URL` em `app/config.py`.

## Ajustes de Modelo e Confiança

Edite `app/config.py`:

```python
MODEL_PATH = BASE_DIR / "runs_screws" / "yolo11_screws" / "weights" / "best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
```

Se o sistema perder parafusos, reduza `CONF_THRESHOLD` para `0.15` ou `0.20`.

Se detectar objetos falsos, aumente para `0.30` ou `0.40`.

Se houver caixas duplicadas no mesmo parafuso, ajuste `IOU_THRESHOLD`.

## Boas Práticas de Foto

Para melhorar a contagem:

- tire a foto em ambiente bem iluminado;
- use fundo branco, cinza claro ou azul liso;
- evite fundo com textura;
- não use flash direto se gerar reflexo;
- espalhe os parafusos sem sobrepor;
- mantenha a câmera paralela à superfície;
- tente manter sempre a mesma distância;
- não corte nenhum parafuso na borda da imagem;
- limpe a lente da câmera;
- confira a imagem antes de enviar.

## Ordem Recomendada de Execução

```bash
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/check_dataset.py --dataset Screw.v4-dataset-screw4.yolov11
python scripts/prepare_dataset.py --input raw_dataset --output dataset
python scripts/convert_segments_to_boxes.py --input Screw.v4-dataset-screw4.yolov11 --output dataset_detect
python scripts/check_dataset.py --dataset dataset_detect
python scripts/train_screws_yolo.py --data dataset_detect/data.yaml --model yolo11s.pt --epochs 100 --imgsz 960
python scripts/validate_screws_yolo.py --data dataset_detect/data.yaml --model runs_screws/yolo11_screws/weights/best.pt
python scripts/count_screws.py --image exemplos/teste.jpg --model runs_screws/yolo11_screws/weights/best.pt --conf 0.25
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Se o dataset já está dividido, pule o `prepare_dataset.py`.
