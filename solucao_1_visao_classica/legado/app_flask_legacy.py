"""
app.py — Aplicação web para testar imagens novas de parafusos.

Como rodar:
    python app.py

Depois acesse:
    http://127.0.0.1:5000

No celular, com computador e celular na mesma rede Wi-Fi:
    http://IP_DO_COMPUTADOR:5000
"""

from __future__ import annotations

import base64
import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from contador_parafusos_web import contar_parafusos_array


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads_testes"
FEEDBACK_DIR = BASE_DIR / "feedback"
UPLOAD_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)

EXTENSOES_PERMITIDAS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _img_to_b64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("Falha ao codificar imagem de saída")
    return base64.b64encode(buf).decode("utf-8")


def _read_upload_to_bgr(file_storage) -> np.ndarray:
    dados = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem enviada")
    return img


HTML = r"""
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Contador de Parafusos — V4 Web</title>
  <style>
    :root{--bg:#0f172a;--card:#111827;--muted:#94a3b8;--text:#e5e7eb;--accent:#22c55e;--accent2:#38bdf8;--danger:#ef4444;}
    *{box-sizing:border-box} body{margin:0;background:linear-gradient(135deg,#020617,#111827 60%,#0f172a);color:var(--text);font-family:Arial,Helvetica,sans-serif;}
    .wrap{max-width:1100px;margin:0 auto;padding:28px 16px 42px}.hero{display:grid;gap:10px;margin-bottom:18px}.hero h1{margin:0;font-size:clamp(1.7rem,5vw,3rem)}.hero p{margin:0;color:var(--muted);line-height:1.5}.panel{background:rgba(17,24,39,.92);border:1px solid rgba(148,163,184,.25);border-radius:20px;padding:18px;box-shadow:0 15px 35px rgba(0,0,0,.25);margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.upload{border:2px dashed rgba(56,189,248,.55);border-radius:18px;padding:20px;text-align:center;background:rgba(2,6,23,.35)}input[type=file]{width:100%;padding:12px;background:#020617;border-radius:12px;border:1px solid rgba(148,163,184,.35);color:var(--text)}label{display:block;margin:12px 0 6px;color:#cbd5e1;font-size:.9rem}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.btn{cursor:pointer;border:0;border-radius:12px;padding:12px 16px;font-weight:700;color:#052e16;background:linear-gradient(135deg,#22c55e,#86efac);box-shadow:0 8px 22px rgba(34,197,94,.25)}.btn2{color:#082f49;background:linear-gradient(135deg,#38bdf8,#bae6fd)}.btn:disabled{opacity:.55;cursor:not-allowed}.hint{font-size:.86rem;color:var(--muted);line-height:1.45}.small{font-size:.78rem;color:var(--muted)}.result-card{background:rgba(2,6,23,.45);border:1px solid rgba(148,163,184,.22);border-radius:18px;padding:14px;margin-top:16px}.result-card h3{margin:0 0 10px}.count{font-size:2.1rem;font-weight:800;color:#22c55e}.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}.imgs img{width:100%;border-radius:12px;border:1px solid rgba(148,163,184,.25);background:#fff}.tag{display:inline-block;border-radius:999px;padding:5px 9px;background:rgba(56,189,248,.13);color:#bae6fd;margin:2px;font-size:.78rem}.err{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);color:#fecaca;border-radius:12px;padding:10px;margin-top:12px}.ok{background:rgba(34,197,94,.11);border:1px solid rgba(34,197,94,.28);color:#bbf7d0;border-radius:12px;padding:10px;margin-top:12px}details{margin-top:10px}summary{cursor:pointer;color:#93c5fd}.footer{color:var(--muted);font-size:.82rem;margin-top:22px;line-height:1.5}
    input[type=number]{width:130px;padding:10px;border-radius:10px;border:1px solid rgba(148,163,184,.35);background:#020617;color:var(--text)}
  </style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>🔩 Contador de Parafusos — V4 Web</h1>
    <p>Envie uma ou várias imagens novas para testar o pipeline V3. A imagem 2, com parafuso único fragmentado, é tratada com agrupamento DBSCAN.</p>
  </section>

  <section class="panel">
    <form action="/processar" method="post" enctype="multipart/form-data">
      <div class="grid">
        <div class="upload">
          <h2>Enviar imagens</h2>
          <input type="file" name="imagens" accept="image/*" multiple required>
          <p class="hint">Pode selecionar várias imagens de uma vez. Para celular, use o botão abaixo para abrir a câmera.</p>
          <input type="file" name="imagens" accept="image/*" capture="environment">
        </div>
        <div>
          <h2>Parâmetros rápidos</h2>
          <label>Área de referência de 1 parafuso</label>
          <input type="number" name="ref_area" value="4021" min="500" max="20000" step="100">
          <p class="hint">Use 4021 para as imagens originais. Se a câmera mudar muito a distância, ajuste esse valor.</p>
          <label>Largura mínima de processamento</label>
          <input type="number" name="min_width" value="640" min="300" max="1600" step="20">
          <p class="hint">Ajuda quando uma imagem nova é muito pequena ou vem da internet.</p>
          <label><input type="checkbox" name="show_rejected"> Mostrar descartados em vermelho</label>
        </div>
      </div>
      <div class="row" style="margin-top:16px">
        <button class="btn" type="submit">Processar imagens</button>
        <a class="btn btn2" href="/api-info" style="text-decoration:none">Ver API</a>
      </div>
    </form>
    <p class="small">Observação: use fotos originais, sem caixas verdes ou texto de contagem desenhados por outro sistema.</p>
  </section>

  {% if erros %}
    <section class="panel">
      <h2>Erros</h2>
      {% for e in erros %}<div class="err">{{ e }}</div>{% endfor %}
    </section>
  {% endif %}

  {% if resultados %}
    <section class="panel">
      <h2>Resultados</h2>
      {% for r in resultados %}
        <div class="result-card">
          <h3>{{ r.nome }}</h3>
          <div>Contagem estimada: <span class="count">{{ r.contagem }}</span></div>
          <div>
            <span class="tag">ref_area usada: {{ r.meta.ref_area_used }}</span>
            <span class="tag">escala: {{ '%.2f'|format(r.meta.scale_applied) }}</span>
            <span class="tag">fragmentação: {{ r.meta.fragmentacao_detectada }}</span>
            <span class="tag">watershed: {{ r.meta.watershed_aplicado }}</span>
          </div>
          <div class="imgs">
            <div><p class="small">Original</p><img src="data:image/jpeg;base64,{{ r.original_b64 }}"></div>
            <div><p class="small">Máscara</p><img src="data:image/jpeg;base64,{{ r.mask_b64 }}"></div>
            <div><p class="small">Resultado</p><img src="data:image/jpeg;base64,{{ r.resultado_b64 }}"></div>
          </div>
          <details>
            <summary>Ver imagem realçada e metadados</summary>
            <div class="imgs">
              <div><p class="small">Imagem realçada</p><img src="data:image/jpeg;base64,{{ r.enhanced_b64 }}"></div>
            </div>
            <pre class="small">{{ r.meta }}</pre>
          </details>
        </div>
      {% endfor %}
    </section>
  {% endif %}

  <section class="footer">
    <b>Uso no relatório:</b> esta aplicação simula o cenário em que o colaborador abre uma página no smartphone, tira/envia uma foto e recebe a contagem. O sistema ainda é visão clássica; a etapa YOLO supervisionada entra depois, quando houver imagens anotadas.
  </section>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, resultados=None, erros=None)


@app.route("/processar", methods=["POST"])
def processar():
    resultados: list[dict[str, Any]] = []
    erros: list[str] = []

    try:
        ref_area = float(request.form.get("ref_area", "4021"))
    except Exception:
        ref_area = 4021.0
    try:
        min_width = int(request.form.get("min_width", "640"))
    except Exception:
        min_width = 640
    show_rejected = request.form.get("show_rejected") == "on"

    arquivos = request.files.getlist("imagens")
    if not arquivos:
        erros.append("Nenhuma imagem enviada.")

    for arquivo in arquivos:
        if not arquivo or arquivo.filename == "":
            continue
        ext = Path(arquivo.filename).suffix.lower().lstrip(".")
        if ext not in EXTENSOES_PERMITIDAS:
            erros.append(f"{arquivo.filename}: extensão não suportada.")
            continue

        try:
            img = _read_upload_to_bgr(arquivo)
            contagem, saida, debug, meta = contar_parafusos_array(
                img, debug=True,
                params={"ref_area": ref_area, "min_width": min_width, "show_rejected": show_rejected},
            )
            resultados.append({
                "nome": arquivo.filename,
                "contagem": contagem,
                "original_b64": _img_to_b64(debug["original"]),
                "mask_b64": _img_to_b64(debug["mask"]),
                "enhanced_b64": _img_to_b64(debug["enhanced"]),
                "resultado_b64": _img_to_b64(saida),
                "meta": meta,
            })
        except Exception as exc:
            erros.append(f"{arquivo.filename}: {exc}")

    return render_template_string(HTML, resultados=resultados, erros=erros)


@app.route("/contar", methods=["POST"])
def contar_api():
    """Endpoint JSON para integração futura com outro front-end/app."""
    if "imagem" not in request.files:
        return jsonify({"erro": "Envie o arquivo no campo 'imagem'."}), 400
    arquivo = request.files["imagem"]
    try:
        img = _read_upload_to_bgr(arquivo)
        contagem, saida, debug, meta = contar_parafusos_array(img, debug=True)
        return jsonify({
            "contagem": contagem,
            "imagem_b64": _img_to_b64(saida),
            "mascara_b64": _img_to_b64(debug["mask"]),
            "meta": meta,
        })
    except Exception as exc:
        return jsonify({"erro": str(exc)}), 500


@app.route("/api-info")
def api_info():
    return jsonify({
        "endpoint": "/contar",
        "method": "POST",
        "field": "imagem",
        "description": "Retorna contagem, imagem processada em base64, máscara e metadados.",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
