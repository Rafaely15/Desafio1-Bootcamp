from __future__ import annotations

import csv
import shutil
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import config
from app.database import Base, engine, get_db
from app.models import Contagem
from app.services.screw_counter import contar_parafusos

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sistema de Contagem de Parafusos")
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "app" / "templates")


def static_url_for_path(path: str | Path) -> str:
    path = Path(path)
    static_root = config.BASE_DIR / "app" / "static"
    try:
        rel = path.resolve().relative_to(static_root.resolve()).as_posix()
        return f"/static/{rel}"
    except ValueError:
        return str(path)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_exists": config.MODEL_PATH.exists(),
        "model_path": str(config.MODEL_PATH),
        "conf_threshold": config.CONF_THRESHOLD,
    }


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "conf": config.CONF_THRESHOLD})


@app.post("/predict")
def predict(
    request: Request,
    funcionario_nome: str = Form(...),
    imagem: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not config.MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Modelo nao encontrado: {config.MODEL_PATH}")
    suffix = Path(imagem.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Envie uma imagem .jpg, .jpeg, .png ou .webp")

    upload_path = config.UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex}{suffix}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(imagem.file, buffer)

    try:
        result = contar_parafusos(
            image_path=upload_path,
            model_path=config.MODEL_PATH,
            output_dir=config.RESULTS_DIR,
            conf=config.CONF_THRESHOLD,
            funcionario_nome=funcionario_nome.strip(),
            iou=config.IOU_THRESHOLD,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record = Contagem(
        funcionario_nome=funcionario_nome.strip(),
        total_parafusos=result["total_parafusos"],
        confianca_media=result["confianca_media"],
        imagem_original=str(upload_path),
        imagem_processada=result["imagem_resultado"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "registro": record,
            "imagem_original_url": static_url_for_path(record.imagem_original),
            "imagem_processada_url": static_url_for_path(record.imagem_processada),
        },
    )


@app.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    registros = db.query(Contagem).order_by(Contagem.data_hora.desc()).limit(500).all()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "registros": registros, "static_url_for_path": static_url_for_path},
    )


def csv_response(registros: list[Contagem], filename: str) -> StreamingResponse:
    rows = [[
        "id",
        "funcionario_nome",
        "data_hora",
        "total_parafusos",
        "confianca_media",
        "imagem_original",
        "imagem_processada",
        "observacao",
    ]]
    for item in registros:
        rows.append([
            item.id,
            item.funcionario_nome,
            item.data_hora.isoformat(sep=" ", timespec="seconds") if item.data_hora else "",
            item.total_parafusos,
            f"{item.confianca_media:.4f}",
            item.imagem_original,
            item.imagem_processada,
            item.observacao or "",
        ])

    def generate():
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        for row in rows:
            writer.writerow(row)
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    registros = db.query(Contagem).order_by(Contagem.data_hora.asc()).all()
    return csv_response(registros, "contagens_parafusos.csv")


@app.get("/export/csv/today")
def export_csv_today(db: Session = Depends(get_db)):
    today = date.today()
    registros = db.query(Contagem).order_by(Contagem.data_hora.asc()).all()
    registros = [r for r in registros if r.data_hora and r.data_hora.date() == today]
    return csv_response(registros, f"contagens_parafusos_{today.isoformat()}.csv")


@app.get("/uploads/{filename}")
def uploads(filename: str):
    path = config.UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo nao encontrado")
    return FileResponse(path)


@app.get("/results/{filename}")
def results(filename: str):
    path = config.RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo nao encontrado")
    return FileResponse(path)


@app.get("/favicon.ico")
def favicon():
    return RedirectResponse(url="/static/favicon.ico")
