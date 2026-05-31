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

    page_w, page_h = 1600, 1067
    margin = 72
    navy = "#062038"
    navy_2 = "#0b3f63"
    copper = "#d79a3d"
    paper = "#fffdf7"
    line_color = "#e4d4bd"
    text_color = "#102333"
    muted = "#66717f"
    card_shadow = "#d8d1c6"

    def font(name: str, size: int):
        path = Path("C:/Windows/Fonts") / name
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            return ImageFont.load_default()

    title_font = font("arialbd.ttf", 62)
    subtitle_font = font("arial.ttf", 28)
    h_font = font("arialbd.ttf", 28)
    small_font = font("arial.ttf", 18)
    small_bold = font("arialbd.ttf", 18)
    table_font = font("arial.ttf", 20)
    table_bold = font("arialbd.ttf", 20)
    big_num_font = font("arialbd.ttf", 44)

    table_columns = [
        ("ID", 95), ("Funcionario", 225), ("Matricula", 175),
        ("Setor", 185), ("Pedido", 180), ("Hora", 150),
        ("Auto", 135), ("Final", 135), ("Status", 205),
    ]
    table_x = margin
    table_w = sum(w for _, w in table_columns)
    row_h = 64
    first_page_rows = 8
    other_page_rows = 10

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

    report_date = records[0].data_hora.date() if records and records[0].data_hora else date.today()

    def draw_brand_mark(draw, x, y, scale=1.0):
        w, h = int(118 * scale), int(132 * scale)
        pts = [
            (x + w // 2, y),
            (x + w, y + h // 4),
            (x + w, y + h * 3 // 4),
            (x + w // 2, y + h),
            (x, y + h * 3 // 4),
            (x, y + h // 4),
        ]
        draw.line(pts + [pts[0]], fill=copper, width=max(2, int(4 * scale)))
        cx = x + w // 2
        draw.rounded_rectangle([cx - 18 * scale, y + 28 * scale, cx + 18 * scale, y + 98 * scale], radius=int(4 * scale), outline=copper, width=max(2, int(3 * scale)))
        draw.rounded_rectangle([cx - 32 * scale, y + 28 * scale, cx + 32 * scale, y + 48 * scale], radius=int(8 * scale), outline=copper, width=max(2, int(3 * scale)))
        for i in range(4):
            yy = y + int((58 + i * 12) * scale)
            draw.line([cx - 24 * scale, yy, cx + 24 * scale, yy + 10 * scale], fill=copper, width=max(2, int(3 * scale)))
        draw.line([x + 24 * scale, y + 58 * scale, x + 36 * scale, y + 68 * scale, x + 28 * scale, y + 78 * scale], fill=copper, width=max(2, int(2 * scale)))
        draw.line([x + w - 24 * scale, y + 58 * scale, x + w - 36 * scale, y + 68 * scale, x + w - 28 * scale, y + 78 * scale], fill=copper, width=max(2, int(2 * scale)))

    def draw_metric_icon(draw, kind, x, y):
        draw.ellipse([x, y, x + 90, y + 90], fill="#f6efe4")
        cx, cy = x + 45, y + 45
        if kind == "registros":
            draw.rounded_rectangle([cx - 18, cy - 24, cx + 18, cy + 24], radius=5, outline="#a87418", width=4)
            draw.rounded_rectangle([cx - 10, cy - 31, cx + 10, cy - 19], radius=4, outline="#a87418", width=4)
            for off in (-10, 2, 14):
                draw.line([cx - 9, cy + off, cx + 11, cy + off], fill="#a87418", width=3)
        elif kind == "automatico":
            draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], outline="#a87418", width=4)
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline="#a87418", width=4)
            for dx, dy in ((0, -32), (0, 32), (-32, 0), (32, 0), (23, 23), (-23, 23), (23, -23), (-23, -23)):
                draw.line([cx + dx * .75, cy + dy * .75, cx + dx, cy + dy], fill="#a87418", width=4)
        elif kind == "final":
            draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], outline="#a87418", width=4)
            draw.line([cx - 12, cy + 1, cx - 2, cy + 11, cx + 15, cy - 12], fill="#a87418", width=4)
        else:
            draw.line([cx - 18, cy + 22, cx + 22, cy - 18], fill="#a87418", width=5)
            draw.polygon([(cx - 24, cy + 28), (cx - 8, cy + 22), (cx - 18, cy + 12)], outline="#a87418")
            draw.rounded_rectangle([cx + 14, cy - 28, cx + 31, cy - 10], radius=5, outline="#a87418", width=4)

    def draw_header(draw, page_no):
        draw.rectangle([0, 0, page_w, 205], fill=navy)
        draw.rectangle([0, 0, page_w, 205], fill=navy_2)
        draw.rectangle([0, 0, page_w, 205], fill=navy)
        draw_brand_mark(draw, margin + 10, 36, 1.0)
        draw.text((margin + 175, 58), "Metal Mec\u00e2nica", fill="white", font=title_font)
        draw.text((margin + 178, 130), title, fill="#ffffff", font=subtitle_font)
        draw.text((page_w - 165, 102), f"P\u00e1gina {page_no}", fill="white", font=small_bold)
        draw.line([page_w - 175, 136, page_w - 68, 136], fill=copper, width=3)
        draw.line([0, 205, page_w, 205], fill=copper, width=4)
        return 258

    def draw_summary(draw, y):
        draw.text((margin + 52, y + 4), f"Data: {report_date.strftime('%d/%m/%Y')}", fill=text_color, font=h_font)
        draw.rounded_rectangle([margin + 10, y, margin + 42, y + 32], radius=4, outline=text_color, width=3)
        draw.line([margin + 10, y + 11, margin + 42, y + 11], fill=text_color, width=3)
        draw.line([margin + 18, y - 6, margin + 18, y + 6], fill=text_color, width=3)
        draw.line([margin + 34, y - 6, margin + 34, y + 6], fill=text_color, width=3)
        y += 76
        cards = [
            ("Registros", summary["registros"], "registros"),
            ("Autom\u00e1tico", summary["automatico"], "automatico"),
            ("Final", summary["final"], "final"),
            ("Corre\u00e7\u00f5es", summary["correcoes"], "correcoes"),
        ]
        card_gap = 30
        card_w = (table_w - card_gap * 3) // 4
        for idx, (label, value, icon) in enumerate(cards):
            x = margin + idx * (card_w + card_gap)
            draw.rounded_rectangle([x + 4, y + 8, x + card_w + 4, y + 138], radius=12, fill=card_shadow)
            draw.rounded_rectangle([x, y, x + card_w, y + 132], radius=12, fill=paper, outline="#d5b784", width=2)
            draw_metric_icon(draw, icon, x + 22, y + 28)
            draw.text((x + 132, y + 42), label, fill=text_color, font=small_bold)
            draw.text((x + 132, y + 84), str(value), fill=navy_2, font=big_num_font)
        return y + 184

    def draw_table(draw, y, rows):
        draw.rounded_rectangle([table_x, y, table_x + table_w, y + row_h], radius=10, fill=navy)
        x = table_x
        for label, width in table_columns:
            draw.text((x + 12, y + 20), label, fill="white", font=table_bold)
            if x > table_x:
                draw.line([x, y + 10, x, y + row_h - 10], fill="#496b83", width=1)
            x += width
        y += row_h
        if not rows:
            draw.rectangle([table_x, y, table_x + table_w, y + row_h], fill=paper, outline=line_color)
            draw.text((table_x + 16, y + 20), "Nenhum registro encontrado.", fill=text_color, font=table_font)
            return y + row_h
        for idx, row in enumerate(rows):
            fill = "#ffffff" if idx % 2 == 0 else "#fbf7ef"
            draw.rectangle([table_x, y, table_x + table_w, y + row_h], fill=fill, outline=line_color)
            x = table_x
            for value, (_, width) in zip(row, table_columns):
                draw.text((x + 14, y + 20), truncate(draw, value, width - 24, table_font), fill=text_color, font=table_font)
                x += width
            y += row_h
        draw.rounded_rectangle([table_x, y - row_h * len(rows) - row_h, table_x + table_w, y], radius=10, outline="#d5b784", width=1)
        return y

    def draw_footer(draw):
        footer_y = page_h - 102
        draw.line([margin, footer_y, page_w - margin, footer_y], fill=copper, width=2)
        draw_brand_mark(draw, margin, footer_y + 18, .42)
        draw.text((margin + 72, footer_y + 36), "Metal Mec\u00e2nica", fill=text_color, font=small_bold)
        draw.text((page_w - 365, footer_y + 42), "C O N T A G E M   D E   P A R A F U S O S", fill=navy, font=small_font)
        draw.ellipse([page_w - 88, footer_y + 38, page_w - 76, footer_y + 50], fill=copper)

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
        draw_footer(draw)
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
