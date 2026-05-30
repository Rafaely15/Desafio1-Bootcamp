from __future__ import annotations

import csv
import json
import shutil
import socket
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import config
from app.database import Base, engine, get_db
from app.models import Contagem
from app.services.screw_counter import contar_parafusos

Base.metadata.create_all(bind=engine)


def ensure_schema() -> None:
    columns = {
        "funcionario_id": "VARCHAR(80)",
        "setor": "VARCHAR(120)",
        "pedido": "VARCHAR(120)",
        "total_corrigido": "INTEGER",
        "status": "VARCHAR(40)",
    }
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(contagens)"))}
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE contagens ADD COLUMN {name} {sql_type}"))


ensure_schema()

app = FastAPI(title="Metal Mecânica - Sistema de Contagem de Parafusos")
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "app" / "templates")


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def static_url_for_path(path: str | Path) -> str:
    path = Path(path)
    static_root = config.BASE_DIR / "app" / "static"
    try:
        rel = path.resolve().relative_to(static_root.resolve()).as_posix()
        return f"/static/{rel}"
    except ValueError:
        return str(path)


def get_employee_from_cookie(request: Request) -> dict[str, str] | None:
    name = request.cookies.get("funcionario_nome", "").strip()
    employee_id = request.cookies.get("funcionario_id", "").strip()
    sector = request.cookies.get("setor", "").strip()
    if not name or not employee_id:
        return None
    return {"nome": name, "id": employee_id, "setor": sector}


def employee_records_today(db: Session, employee: dict[str, str] | None) -> list[Contagem]:
    if not employee:
        return []
    today = date.today()
    records = (
        db.query(Contagem)
        .filter(Contagem.funcionario_id == employee["id"])
        .order_by(Contagem.data_hora.desc())
        .all()
    )
    return [item for item in records if item.data_hora and item.data_hora.date() == today]


def employee_summary(records: list[Contagem]) -> dict[str, int]:
    total_records = len(records)
    total_screws = sum(
        int(item.total_corrigido if item.total_corrigido is not None else item.total_parafusos)
        for item in records
    )
    corrections = sum(1 for item in records if item.status == "corrigida")
    return {"registros": total_records, "parafusos": total_screws, "correcoes": corrections}


def records_for_date(db: Session, target_date: date) -> list[Contagem]:
    records = db.query(Contagem).order_by(Contagem.data_hora.asc()).all()
    return [item for item in records if item.data_hora and item.data_hora.date() == target_date]


def day_summary(records: list[Contagem]) -> dict[str, int]:
    total_records = len(records)
    total_auto = sum(int(item.total_parafusos) for item in records)
    total_final = sum(
        int(item.total_corrigido if item.total_corrigido is not None else item.total_parafusos)
        for item in records
    )
    corrections = sum(1 for item in records if item.status == "corrigida")
    return {
        "registros": total_records,
        "automatico": total_auto,
        "final": total_final,
        "correcoes": corrections,
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_exists": config.MODEL_PATH.exists(),
        "model_path": str(config.MODEL_PATH),
        "conf_threshold": config.CONF_THRESHOLD,
    }


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    employee = get_employee_from_cookie(request)
    records = employee_records_today(db, employee)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "conf": config.CONF_THRESHOLD,
            "local_ip": get_local_ip(),
            "employee": employee,
            "records_today": records,
            "summary": employee_summary(records),
        },
    )


@app.post("/login")
def login(
    funcionario_nome: str = Form(...),
    funcionario_id: str = Form(...),
    setor: str = Form(""),
):
    if not funcionario_nome.strip() or not funcionario_id.strip():
        raise HTTPException(status_code=400, detail="Informe nome e matricula/ID.")
    response = RedirectResponse(url="/", status_code=303)
    max_age = 60 * 60 * 12
    response.set_cookie("funcionario_nome", funcionario_nome.strip(), max_age=max_age, samesite="lax")
    response.set_cookie("funcionario_id", funcionario_id.strip(), max_age=max_age, samesite="lax")
    response.set_cookie("setor", setor.strip(), max_age=max_age, samesite="lax")
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse(url="/", status_code=303)
    for key in ("funcionario_nome", "funcionario_id", "setor"):
        response.delete_cookie(key)
    return response


@app.post("/predict")
def predict(
    request: Request,
    pedido: str = Form(""),
    observacao: str = Form(""),
    imagem: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    employee = get_employee_from_cookie(request)
    if not employee:
        return RedirectResponse(url="/", status_code=303)
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
            funcionario_id=employee["id"],
            funcionario_nome=employee["nome"],
            iou=config.IOU_THRESHOLD,
            imgsz=config.DEFAULT_IMGSZ,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record = Contagem(
        funcionario_nome=employee["nome"],
        funcionario_id=employee["id"],
        setor=employee["setor"],
        pedido=pedido.strip(),
        total_parafusos=result["total_parafusos"],
        total_corrigido=result["total_parafusos"],
        confianca_media=result["confianca_media"],
        imagem_original=str(upload_path),
        imagem_processada=result["imagem_resultado"],
        status="confirmada",
        observacao=observacao.strip(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "registro": record,
            "imagem_original_url": static_url_for_path(record.imagem_original),
            "imagem_processada_url": static_url_for_path(record.imagem_processada),
        },
    )


@app.post("/record/{record_id}/correct")
def correct_record(
    record_id: int,
    total_corrigido: int = Form(...),
    observacao: str = Form(""),
    db: Session = Depends(get_db),
):
    record = db.query(Contagem).filter(Contagem.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Registro nao encontrado")
    record.total_corrigido = int(total_corrigido)
    record.status = "corrigida" if int(total_corrigido) != record.total_parafusos else "confirmada"
    if observacao.strip():
        record.observacao = observacao.strip()
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    registros = db.query(Contagem).order_by(Contagem.data_hora.desc()).limit(500).all()
    today_records = records_for_date(db, date.today())
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "registros": registros,
            "static_url_for_path": static_url_for_path,
            "today_summary": day_summary(today_records),
            "finalizado": request.query_params.get("finalizado", ""),
        },
    )


@app.post("/finalizar-dia")
def finalizar_dia(db: Session = Depends(get_db)):
    target_date = date.today()
    records = records_for_date(db, target_date)
    summary = day_summary(records)
    output_dir = config.BASE_DIR / "outputs" / "fechamentos"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    filename = f"fechamento_{target_date.isoformat()}_{stamp}.json"
    csv_filename = f"fechamento_{target_date.isoformat()}_{stamp}.csv"
    payload = {
        "empresa": "Metal Mecânica",
        "data": target_date.isoformat(),
        "registrado_em": datetime.now().isoformat(timespec="seconds"),
        "resumo": summary,
        "registros": [item.id for item in records],
    }
    (output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / csv_filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "id",
            "funcionario_nome",
            "funcionario_id",
            "setor",
            "pedido",
            "data_hora",
            "total_parafusos",
            "total_corrigido",
            "confianca_media",
            "status",
            "imagem_original",
            "imagem_processada",
            "observacao",
        ])
        for item in records:
            writer.writerow([
                item.id,
                item.funcionario_nome,
                item.funcionario_id or "",
                item.setor or "",
                item.pedido or "",
                item.data_hora.isoformat(sep=" ", timespec="seconds") if item.data_hora else "",
                item.total_parafusos,
                item.total_corrigido if item.total_corrigido is not None else item.total_parafusos,
                f"{item.confianca_media:.4f}",
                item.status or "",
                item.imagem_original,
                item.imagem_processada,
                item.observacao or "",
            ])

    for item in records:
        db.delete(item)
    db.commit()

    return RedirectResponse(url=f"/dashboard?finalizado={csv_filename}", status_code=303)


def csv_response(registros: list[Contagem], filename: str) -> StreamingResponse:
    rows = [[
        "id",
        "funcionario_nome",
        "funcionario_id",
        "setor",
        "pedido",
        "data_hora",
        "total_parafusos",
        "total_corrigido",
        "confianca_media",
        "status",
        "imagem_original",
        "imagem_processada",
        "observacao",
    ]]
    for item in registros:
        rows.append([
            item.id,
            item.funcionario_nome,
            item.funcionario_id or "",
            item.setor or "",
            item.pedido or "",
            item.data_hora.isoformat(sep=" ", timespec="seconds") if item.data_hora else "",
            item.total_parafusos,
            item.total_corrigido if item.total_corrigido is not None else item.total_parafusos,
            f"{item.confianca_media:.4f}",
            item.status or "",
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
