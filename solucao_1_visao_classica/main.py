from __future__ import annotations

import csv
import json
import shutil
import socket
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import config
from app.database import Base, engine, get_db
from app.models import Contagem
from contador_parafusos_web import contar_parafusos_array

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
    total_auto = sum(int(item.total_parafusos) for item in records)
    total_final = sum(
        int(item.total_corrigido if item.total_corrigido is not None else item.total_parafusos)
        for item in records
    )
    corrections = sum(1 for item in records if item.status == "corrigida")
    return {
        "registros": len(records),
        "automatico": total_auto,
        "final": total_final,
        "correcoes": corrections,
    }


def group_records_by_day(records: list[Contagem]) -> list[dict[str, object]]:
    grouped: dict[date, list[Contagem]] = {}
    for item in records:
        if item.data_hora:
            grouped.setdefault(item.data_hora.date(), []).append(item)
    return [
        {"date": day, "records": grouped[day], "summary": day_summary(grouped[day])}
        for day in sorted(grouped.keys(), reverse=True)
    ]


def fechamento_rows(records: list[Contagem]) -> list[list[str]]:
    rows = [[
        "id", "funcionario_nome", "funcionario_id", "setor", "pedido",
        "data_hora", "total_parafusos", "total_corrigido", "confianca_media",
        "status", "imagem_original", "imagem_processada", "observacao",
    ]]
    for item in records:
        rows.append([
            str(item.id),
            item.funcionario_nome,
            item.funcionario_id or "",
            item.setor or "",
            item.pedido or "",
            item.data_hora.isoformat(sep=" ", timespec="seconds") if item.data_hora else "",
            str(item.total_parafusos),
            str(item.total_corrigido if item.total_corrigido is not None else item.total_parafusos),
            f"{item.confianca_media:.4f}",
            item.status or "",
            item.imagem_original,
            item.imagem_processada,
            item.observacao or "",
        ])
    return rows


def build_pdf_report(title: str, records: list[Contagem], summary: dict[str, int]) -> bytes:
    import io
    from PIL import Image, ImageDraw, ImageFont

    page_w, page_h = 1240, 1754
    margin = 72
    navy = "#173c56"
    blue = "#1f5f8f"
    paper = "#fffdf7"
    line_color = "#e4d4bd"
    text_color = "#102333"
    muted = "#6d5744"

    def font(name: str, size: int):
        path = Path("C:/Windows/Fonts") / name
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            return ImageFont.load_default()

    title_font = font("arialbd.ttf", 42)
    subtitle_font = font("arial.ttf", 22)
    h_font = font("arialbd.ttf", 24)
    small_font = font("arial.ttf", 18)
    table_font = font("arial.ttf", 17)
    table_bold = font("arialbd.ttf", 17)
    big_num_font = font("arialbd.ttf", 38)

    table_columns = [
        ("ID", 55), ("Funcionario", 220), ("Matricula", 120),
        ("Setor", 130), ("Pedido", 105), ("Hora", 100),
        ("Auto", 70), ("Final", 70), ("Status", 195),
    ]
    table_x = margin
    table_w = sum(w for _, w in table_columns)
    row_h = 38
    first_page_rows = 24
    other_page_rows = 32

    data = []
    for item in records:
        final = item.total_corrigido if item.total_corrigido is not None else item.total_parafusos
        data.append([
            str(item.id), item.funcionario_nome, item.funcionario_id or "-",
            item.setor or "-", item.pedido or "-",
            item.data_hora.strftime("%H:%M:%S") if item.data_hora else "-",
            str(item.total_parafusos), str(final), item.status or "confirmada",
        ])

    def truncate(draw, value, max_w, fnt):
        value = str(value)
        if draw.textlength(value, font=fnt) <= max_w:
            return value
        while value and draw.textlength(value + "...", font=fnt) > max_w:
            value = value[:-1]
        return value + "..."

    def draw_header(draw, page_no):
        draw.rounded_rectangle([0, 0, page_w, 150], radius=0, fill=navy)
        draw.text((margin, 36), "Metal Mecanica", fill="white", font=title_font)
        draw.text((margin, 91), title, fill="#dceaf3", font=subtitle_font)
        draw.text((page_w - 210, 48), f"Pagina {page_no}", fill="#dceaf3", font=small_font)
        return 190

    def draw_summary(draw, y):
        draw.text((margin, y), f"Data: {date.today().strftime('%d/%m/%Y')}", fill=text_color, font=h_font)
        y += 48
        cards = [
            ("Registros", summary["registros"]),
            ("Automatico", summary["automatico"]),
            ("Final", summary["final"]),
            ("Correcoes", summary["correcoes"]),
        ]
        card_w = (table_w - 36) // 4
        for idx, (label, value) in enumerate(cards):
            x = margin + idx * (card_w + 12)
            draw.rounded_rectangle([x, y, x + card_w, y + 112], radius=16, fill=paper, outline=line_color, width=2)
            draw.text((x + 22, y + 18), label, fill=muted, font=small_font)
            draw.text((x + 22, y + 48), str(value), fill=blue, font=big_num_font)
        return y + 154

    def draw_table(draw, y, rows):
        draw.rounded_rectangle([table_x, y, table_x + table_w, y + row_h], radius=8, fill=navy)
        x = table_x
        for label, width in table_columns:
            draw.text((x + 8, y + 10), label, fill="white", font=table_bold)
            x += width
        y += row_h
        if not rows:
            draw.rectangle([table_x, y, table_x + table_w, y + row_h], fill=paper, outline=line_color)
            draw.text((table_x + 12, y + 10), "Nenhum registro encontrado.", fill=text_color, font=table_font)
            return y + row_h
        for idx, row in enumerate(rows):
            fill = "#ffffff" if idx % 2 == 0 else "#f7efe3"
            draw.rectangle([table_x, y, table_x + table_w, y + row_h], fill=fill, outline=line_color)
            x = table_x
            for value, (_, width) in zip(row, table_columns):
                draw.text((x + 8, y + 10), truncate(draw, value, width - 16, table_font), fill=text_color, font=table_font)
                x += width
            y += row_h
        return y

    pages = []
    remaining = data[:]
    page_no = 1
    while remaining or not pages:
        image = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(image)
        y = draw_header(draw, page_no)
        rows_per_page = other_page_rows
        if page_no == 1:
            y = draw_summary(draw, y)
            rows_per_page = first_page_rows
        rows = remaining[:rows_per_page]
        remaining = remaining[rows_per_page:]
        draw_table(draw, y, rows)
        draw.line([margin, page_h - 58, page_w - margin, page_h - 58], fill=line_color, width=2)
        draw.text(
            (margin, page_h - 42),
            "Relatorio gerado automaticamente pelo Sistema de Contagem de Parafusos.",
            fill=muted,
            font=small_font,
        )
        pages.append(image)
        page_no += 1

    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "method": "visao_classica_v5",
        "description": "Pipeline OpenCV com segmentacao adaptativa e watershed",
    }


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    employee = get_employee_from_cookie(request)
    records = employee_records_today(db, employee)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "conf": 0.91,
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

    suffix = Path(imagem.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Envie uma imagem .jpg, .jpeg, .png ou .webp")

    upload_path = config.UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex}{suffix}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(imagem.file, buffer)

    try:
        img_data = np.frombuffer(upload_path.read_bytes(), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Não foi possível decodificar a imagem.")

        count, result_img, _, meta = contar_parafusos_array(img_bgr, debug=False)

        result_filename = f"result_{upload_path.stem}.jpg"
        result_path = config.RESULTS_DIR / result_filename
        cv2.imwrite(str(result_path), result_img)

        confianca = 0.72 if meta.get("low_confidence", False) else 0.91
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record = Contagem(
        funcionario_nome=employee["nome"],
        funcionario_id=employee["id"],
        setor=employee["setor"],
        pedido=pedido.strip(),
        total_parafusos=count,
        total_corrigido=count,
        confianca_media=confianca,
        imagem_original=str(upload_path),
        imagem_processada=str(result_path),
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
        raise HTTPException(status_code=404, detail="Registro não encontrado")
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
            "grouped_records": group_records_by_day(registros),
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
    csv_filename = f"fechamento_{target_date.isoformat()}_{stamp}.csv"
    pdf_filename = f"fechamento_{target_date.isoformat()}_{stamp}.pdf"
    payload = {
        "empresa": "Metal Mecânica",
        "data": target_date.isoformat(),
        "registrado_em": datetime.now().isoformat(timespec="seconds"),
        "resumo": summary,
        "registros": [item.id for item in records],
    }
    json_filename = f"fechamento_{target_date.isoformat()}_{stamp}.json"
    (output_dir / json_filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / csv_filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(fechamento_rows(records))

    (output_dir / pdf_filename).write_bytes(
        build_pdf_report("Fechamento do dia - Contagem de Parafusos", records, summary)
    )

    return RedirectResponse(url=f"/dashboard?finalizado={csv_filename} e {pdf_filename}", status_code=303)


def csv_response(registros: list[Contagem], filename: str) -> StreamingResponse:
    rows = [[
        "id", "funcionario_nome", "funcionario_id", "setor", "pedido",
        "data_hora", "total_parafusos", "total_corrigido", "confianca_media",
        "status", "imagem_original", "imagem_processada", "observacao",
    ]]
    for item in registros:
        rows.append([
            item.id, item.funcionario_nome, item.funcionario_id or "",
            item.setor or "", item.pedido or "",
            item.data_hora.isoformat(sep=" ", timespec="seconds") if item.data_hora else "",
            item.total_parafusos,
            item.total_corrigido if item.total_corrigido is not None else item.total_parafusos,
            f"{item.confianca_media:.4f}", item.status or "",
            item.imagem_original, item.imagem_processada, item.observacao or "",
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


@app.get("/export/pdf/today")
def export_pdf_today(db: Session = Depends(get_db)):
    today = date.today()
    registros = records_for_date(db, today)
    pdf = build_pdf_report("Relatorio do dia - Contagem de Parafusos", registros, day_summary(registros))
    filename = f"contagens_parafusos_{today.isoformat()}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/favicon.ico")
def favicon():
    return RedirectResponse(url="/static/favicon.ico")
