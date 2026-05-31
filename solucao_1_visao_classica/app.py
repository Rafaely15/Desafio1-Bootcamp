"""Aplicacao Streamlit orientada ao uso operacional no picking."""

from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from database_utils import (
    export_daily_csv,
    export_employee_csv,
    get_daily_summary,
    get_record_dirs,
    init_db,
    list_employees,
    load_records_by_date,
    load_records_by_employee_and_date,
    sanitize_filename,
    save_count_record,
)
from screw_counter import count_screws_for_app


ROOT = Path(__file__).resolve().parent
ADMIN_PASSWORD = "admin123"
EXTS = ["jpg", "jpeg", "png", "webp", "bmp"]
GUIDE_IMAGE = ROOT / "assets" / "capture_guide.png"


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #1C3448;
            --navy-dark: #102333;
            --blue: #1F5F8F;
            --copper: #C99566;
            --gold: #B67A2D;
            --cream: #FFF7E9;
            --paper: #FFFCF4;
            --border: #D7C7AD;
            --muted: #6D5744;
        }
        .stApp {
            background:
                radial-gradient(circle at 80% 8%, rgba(201, 149, 102, .22), transparent 28rem),
                linear-gradient(110deg, #FFF8EA 0%, #FFFFFF 48%, #F7EAD5 100%);
            color: var(--navy);
        }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--navy);
        }
        .brandbar,
        .brandbar div,
        .brandbar span,
        .brandbar h1 {
            color: white;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-testid="stFileUploaderDropzone"] {
            background: #FFFDF7 !important;
            color: var(--navy) !important;
            border: 2px solid #9C6B48 !important;
            border-radius: 10px !important;
            box-shadow: inset 0 1px 5px rgba(112, 70, 38, .18);
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
        }
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] p {
            color: #F2C7A6 !important;
        }
        div[role="radiogroup"] label,
        div[role="radiogroup"] label p,
        div[role="radiogroup"] span {
            color: var(--navy) !important;
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 250, 238, .98), rgba(253, 240, 213, .96));
            border-right: 1px solid #D4BE9B;
            box-shadow: 16px 0 36px rgba(54, 38, 20, .08);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }
        .brandbar {
            background: linear-gradient(135deg, #20384D 0%, #2D5578 100%);
            margin: -1.25rem calc(50% - 50vw) 1.3rem;
            padding: 1rem max(1.2rem, calc(50vw - 590px));
            box-shadow: 0 12px 34px rgba(16, 35, 51, .24);
            display: flex;
            align-items: center;
            gap: .85rem;
        }
        .brandmark {
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            display: inline-grid;
            place-items: center;
            background: radial-gradient(circle, #F4D0AD 0%, #A46B3A 52%, #183248 100%);
            color: #0D2233 !important;
            font-weight: 900;
            border: 2px solid rgba(255,255,255,.55);
            box-shadow: 0 5px 16px rgba(0,0,0,.3);
        }
        .brandbar h1 {
            font-size: clamp(1.45rem, 3vw, 2.15rem);
            margin: 0;
            letter-spacing: 0;
        }
        .soft-card,
        .guide-card,
        .result-panel {
            background: rgba(255, 252, 244, .94);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 14px 34px rgba(73, 47, 18, .16);
            margin: .85rem 0;
        }
        .soft-card {
            padding: 1rem;
        }
        .guide-card {
            overflow: hidden;
        }
        .mobile-strip {
            background: rgba(255, 248, 235, .96);
            border-bottom: 1px solid #E6D5BA;
            padding: .78rem 1.1rem;
            display: flex;
            align-items: center;
            gap: .7rem;
            flex-wrap: wrap;
            font-weight: 800;
        }
        .mobile-strip code {
            background: #EBD1B4;
            color: #6F3F21;
            border-radius: 8px;
            padding: .22rem .55rem;
            font-size: 1rem;
        }
        .guide-image {
            padding: .85rem .9rem 0;
        }
        .guide-title {
            text-align: center;
            font-size: clamp(1.35rem, 3vw, 2rem);
            font-weight: 900;
            margin: .25rem 0 .55rem;
            color: #101820;
        }
        .guide-labels {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .65rem;
            padding: .2rem 1rem 1rem;
            text-align: center;
            font-size: clamp(.92rem, 2vw, 1.08rem);
            font-weight: 800;
            color: #101820;
        }
        .capture-actions {
            padding: .25rem 1rem 1rem;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, #193247, #284E6B) !important;
            border: 2px dashed #B98555 !important;
            min-height: 6.4rem;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #E3BE75, #9E681E) !important;
            color: #1B130C !important;
            border: 1px solid #F4DFAF !important;
            border-radius: 10px !important;
            font-weight: 800 !important;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .42rem .7rem;
            border-radius: 999px;
            background: #EAD2B7;
            color: #5B341C;
            font-weight: 700;
            font-size: .95rem;
        }
        .success-count {
            background: linear-gradient(135deg, #F8E5C7 0%, #FFFDF7 100%);
            border: 1px solid #D6B98E;
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 12px 30px rgba(82, 51, 20, .12);
        }
        .success-count h2 {
            margin: 0;
            color: #17527B;
            font-size: clamp(1.55rem, 5vw, 2.5rem);
            letter-spacing: 0;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
            min-height: 3rem;
            font-weight: 750;
            border: 0;
            background: linear-gradient(135deg, #1F3B54 0%, #2D5578 68%, #C99566 135%);
            color: white;
            box-shadow: 0 8px 18px rgba(31, 59, 84, .24);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1F3B54 0%, #2D5578 68%, #C99566 135%);
            color: white;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #1D4F73 0 .45rem, #FFFDF7 .45rem 100%);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.05rem .9rem .85rem;
            box-shadow: 0 8px 24px rgba(73, 47, 18, .13);
        }
        [data-testid="stMetricValue"] {
            color: #17628F;
            font-weight: 900;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        .small-muted {
            color: var(--muted);
            font-size: .92rem;
        }
        .quality-ok {
            background: #ECFDF5;
            border: 1px solid #BBF7D0;
            color: #166534;
            border-radius: 14px;
            padding: .8rem .9rem;
            margin: .75rem 0;
            font-weight: 650;
        }
        .quality-alert {
            background: #FFF7ED;
            border: 1px solid #FED7AA;
            color: #9A3412;
            border-radius: 14px;
            padding: .8rem .9rem;
            margin: .75rem 0;
            font-weight: 650;
        }
        .quality-alert li,
        .quality-alert p,
        .quality-alert div {
            color: #9A3412 !important;
        }
        @media (max-width: 640px) {
            .block-container { padding-left: .85rem; padding-right: .85rem; }
            .brandbar { margin-left: -.85rem; margin-right: -.85rem; padding: .85rem; }
            .soft-card { border-radius: 14px; padding: .9rem; }
            .guide-labels { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_uploaded_to_bgr(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nao foi possivel abrir a imagem enviada.")
    return img


def save_images(employee_id: str, result: dict[str, Any]) -> tuple[Path, Path]:
    now = datetime.now()
    record_date = now.date().isoformat()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_id = sanitize_filename(employee_id)
    dirs = get_record_dirs(record_date)
    original_path = dirs["originals"] / f"{safe_id}_{timestamp}_original.jpg"
    detection_path = dirs["detections"] / f"{safe_id}_{timestamp}_detection.jpg"
    cv2.imwrite(str(original_path), result["original"])
    cv2.imwrite(str(detection_path), result["annotated"])
    return original_path, detection_path


def save_operational_record(status: str, final_count: int) -> int:
    result = st.session_state.last_detection
    employee = st.session_state.employee
    now = datetime.now()
    original_path, detection_path = save_images(employee["employee_id"], result)
    automatic_count = int(result["count"])
    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "employee_name": employee["employee_name"],
        "employee_id": employee["employee_id"],
        "sector": employee.get("sector", ""),
        "order_id": st.session_state.get("current_order_id", ""),
        "automatic_count": automatic_count,
        "final_count": int(final_count),
        "correction": int(final_count) - automatic_count,
        "status": status,
        "image_original_path": str(original_path.relative_to(ROOT)),
        "image_result_path": str(detection_path.relative_to(ROOT)),
        "csv_export_path": "",
        "notes": st.session_state.get("current_notes", ""),
        "metadata_json": json.dumps(result.get("metadata", {}), ensure_ascii=False),
    }
    record_id = save_count_record(record)
    export_daily_csv(record["date"])
    st.session_state.last_saved_id = record_id
    return record_id


def employee_login() -> bool:
    with st.sidebar:
        st.markdown("### Area tecnica / supervisor")
        st.divider()
        st.markdown("### Identificacao do funcionario")

        if "employee" in st.session_state:
            employee = st.session_state.employee
            st.success(f"{employee['employee_name']} - {employee['employee_id']}")
            if st.button("Sair / trocar funcionario"):
                for key in ["employee", "last_detection", "show_correction", "show_admin_debug"]:
                    st.session_state.pop(key, None)
                st.rerun()
            return True

        with st.form("employee_login_form"):
            name = st.text_input("Nome do funcionario")
            employee_id = st.text_input("Matricula ou ID")
            sector = st.text_input("Setor (opcional)")
            submitted = st.form_submit_button("Entrar", type="primary")

        if submitted:
            if not name.strip() or not employee_id.strip():
                st.warning("Informe nome e matricula para continuar.")
                return False
            st.session_state.employee = {
                "employee_name": name.strip(),
                "employee_id": employee_id.strip(),
                "sector": sector.strip(),
            }
            st.session_state.last_detection = None
            st.rerun()

    return "employee" in st.session_state


def render_brand_header() -> None:
    st.markdown(
        """
        <div class="brandbar">
            <span class="brandmark">AI</span>
            <h1>MetaCheck AI - Hardware Counter</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mobile_strip() -> None:
    local_ip = _get_local_ip()
    st.markdown(
        f"""
        <div class="mobile-strip">
            <span>Abrir no celular:</span>
            <code>http://{local_ip}:8501</code>
            <span style="font-weight:600;">(mesma rede Wi-Fi)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_capture_guide() -> None:
    st.markdown('<div class="guide-card">', unsafe_allow_html=True)
    render_mobile_strip()
    st.markdown('<div class="guide-image">', unsafe_allow_html=True)
    st.markdown('<div class="guide-title">Formato ideal</div>', unsafe_allow_html=True)
    if GUIDE_IMAGE.exists():
        st.image(str(GUIDE_IMAGE), use_column_width=True)
    else:
        st.info("Imagem guia nao encontrada em assets/capture_guide.png.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="guide-labels">
            <div>Mantenha a camera nivelada</div>
            <div>Boa iluminacao</div>
            <div>Distancia ideal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_instructions() -> None:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Como tirar uma boa foto")
    st.write("Quanto melhor a foto, mais precisa sera a contagem.")
    st.markdown(
        """
        1. Use um fundo claro e uniforme.
        2. Evite sombras fortes sobre os parafusos.
        3. Tire a foto de cima, com o celular o mais perpendicular possivel.
        4. Separe os parafusos sempre que possivel.
        5. Evite sobrepor muitos parafusos.
        6. Mantenha todos os parafusos dentro da imagem.
        7. Use boa iluminacao.
        8. Se a foto estiver tremida, tire novamente.
        """
    )
    with st.expander("Ver exemplos de boas praticas"):
        st.markdown(
            """
            - Fundo recomendado: claro, sem estampas e sem textura forte.
            - Iluminacao: uniforme, sem reflexos diretos no metal.
            - Distancia da camera: perto o suficiente para ver as pecas, mas sem cortar bordas.
            - Distribuicao: espalhe os parafusos e reduza sobreposicoes quando o processo permitir.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_capture() -> None:
    tab_count, tab_calibrate = st.tabs(["Contar parafusos", "Calibrar camera"])

    # Aba 1: contagem
    with tab_count:
        render_capture_guide()

        active_ref = st.session_state.get("calibrated_ref_area")
        if active_ref:
            st.success(
                f"Calibracao ativa: ref_area = {active_ref:.0f} px2"
            )
        else:
            st.info("Usando ref_area padrao (4021 px2). Calibre na aba Calibrar camera se a contagem estiver errada.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.session_state.current_order_id = st.text_input("Numero do pedido ou lote (opcional)")
        with col_b:
            st.session_state.current_notes = st.text_input("Observacoes (opcional)")

        source = st.radio(
            "Escolha como capturar a imagem",
            ["Tirar foto agora", "Enviar imagem da galeria"],
            horizontal=True,
        )
        image_bgr = None
        if source == "Tirar foto agora":
            camera_file = st.camera_input("Tirar foto")
            if camera_file is not None:
                image_bgr = read_uploaded_to_bgr(camera_file)
        else:
            upload = st.file_uploader("Arraste a imagem ou procure arquivos", type=EXTS)
            if upload is not None:
                image_bgr = read_uploaded_to_bgr(upload)

        if image_bgr is not None:
            with st.spinner("Processando imagem..."):
                calibrated = st.session_state.get("calibrated_ref_area")
                st.session_state.last_detection = count_screws_for_app(
                    image_bgr,
                    ref_area_override=calibrated,
                )

    # Aba 2: calibracao
    with tab_calibrate:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Calibrar para sua camera e distancia")
        st.markdown("""
        Calibre **uma vez** sempre que:
        - Mudar o celular ou a distancia de captura
        - Os parafusos forem de um tipo diferente (maior ou menor)
        - A contagem estiver sistematicamente errada para mais ou para menos

        **Como fazer:**
        1. Coloque **exatamente 1 parafuso** isolado em fundo plano e bem iluminado
        2. Fotografe na **mesma distancia** que usara no trabalho
        3. Toque em "Calibrar" e confirme o resultado
        """)

        cal_source = st.radio(
            "Foto para calibracao:",
            ["Tirar foto agora", "Enviar da galeria"],
            horizontal=True,
            key="cal_source",
        )
        cal_img = None
        if cal_source == "Tirar foto agora":
            cal_file = st.camera_input("Foto de 1 parafuso para calibracao", key="cal_camera")
            if cal_file:
                cal_img = read_uploaded_to_bgr(cal_file)
        else:
            cal_upload = st.file_uploader(
                "Imagem de 1 parafuso isolado", type=EXTS, key="cal_upload"
            )
            if cal_upload:
                cal_img = read_uploaded_to_bgr(cal_upload)

        if cal_img is not None:
            with st.spinner("Calculando area de referencia..."):
                from screw_counter import calibrate_from_single_screw
                cal_result = calibrate_from_single_screw(cal_img)

            conf_color = {"alta": "normal", "media": "off", "baixa": "inverse"}
            conf_emoji = {"alta": "OK", "media": "Atencao", "baixa": "Erro"}
            current_ref = st.session_state.get("calibrated_ref_area", 4021.0)

            st.metric(
                "Ref_area medida",
                f"{cal_result['ref_area']:.0f} px2",
                delta=f"{cal_result['ref_area'] - current_ref:+.0f} vs atual",
                delta_color=conf_color.get(cal_result["confidence"], "off"),
            )
            conf = cal_result["confidence"]
            st.markdown(
                f"**Confianca: {conf.upper()}** — {cal_result['message']}"
            )
            st.image(
                cal_result["debug_image"],
                caption="Parafuso detectado para calibracao",
                use_column_width=True,
            )

            col_apply, col_reset = st.columns(2)
            with col_apply:
                if conf in ("alta", "media"):
                    if st.button("Aplicar calibracao", type="primary"):
                        st.session_state["calibrated_ref_area"] = cal_result["ref_area"]
                        st.success(
                            f"Calibracao aplicada! Proximas contagens usarao "
                            f"ref_area = {cal_result['ref_area']:.0f} px2."
                        )
                        st.rerun()
            with col_reset:
                if st.session_state.get("calibrated_ref_area"):
                    if st.button("Redefinir para padrao"):
                        del st.session_state["calibrated_ref_area"]
                        st.info("Ref_area redefinida para 4021 px2 (padrao).")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_result() -> None:
    result = st.session_state.get("last_detection")
    if not result:
        return
    count = int(result["count"])
    st.markdown('<div class="success-count">', unsafe_allow_html=True)
    st.markdown(f"<h2>Contagem estimada: {count} parafusos</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    quality = result.get("metadata", {}).get("quality", {})
    warnings = quality.get("warnings", []) if isinstance(quality, dict) else []
    if warnings:
        items = "".join(f"<li>{warning}</li>" for warning in warnings)
        st.markdown(
            f'<div class="quality-alert"><strong>Atencao a qualidade da foto:</strong><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="quality-ok">Foto com qualidade adequada para conferencia. Revise a imagem antes de salvar.</div>',
            unsafe_allow_html=True,
        )
    st.image(bgr_to_rgb(result["annotated"]), caption="Imagem analisada", use_column_width=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("A contagem esta correta?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Sim, salvar contagem", type="primary"):
            record_id = save_operational_record("confirmada", count)
            st.success(f"Registro salvo com sucesso. ID {record_id}.")
            st.session_state.last_detection = None
            st.rerun()
    with col_no:
        if st.button("Nao, corrigir manualmente"):
            st.session_state.show_correction = True
    if st.session_state.get("show_correction"):
        corrected = st.number_input("Informe a contagem correta", min_value=0, max_value=9999, value=count, step=1)
        if st.button("Salvar contagem corrigida", type="primary"):
            record_id = save_operational_record("corrigida", int(corrected))
            st.success(f"Correcao salva com sucesso. ID {record_id}.")
            st.session_state.show_correction = False
            st.session_state.last_detection = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_employee_history() -> None:
    employee = st.session_state.employee
    today = datetime.now().date().isoformat()
    df = load_records_by_employee_and_date(employee["employee_id"], today)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Meus registros de hoje")
    if df.empty:
        st.info("Nenhum registro salvo hoje.")
    else:
        total_records = len(df)
        total_screws = int(df["final_count"].fillna(0).sum())
        corrections = int((df["status"] == "corrigida").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Conferencias", total_records)
        c2.metric("Parafusos", total_screws)
        c3.metric("Correcoes", corrections)
        table = df.copy()
        table["horario"] = pd.to_datetime(table["timestamp"]).dt.strftime("%H:%M:%S")
        st.dataframe(
            table[["horario", "order_id", "automatic_count", "final_count", "status"]].rename(
                columns={
                    "order_id": "pedido/lote",
                    "automatic_count": "automatica",
                    "final_count": "final",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    col_day, col_me = st.columns(2)
    daily_csv = export_daily_csv(today)
    employee_csv = export_employee_csv(employee["employee_id"])
    with col_day:
        st.download_button("⬇️ Baixar CSV do dia", daily_csv.read_bytes(), daily_csv.name, "text/csv")
    with col_me:
        st.download_button("⬇️ Baixar meus registros", employee_csv.read_bytes(), employee_csv.name, "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


def render_admin_area() -> None:
    with st.sidebar.expander("🔐 Area tecnica / supervisor"):
        password = st.text_input("Senha", type="password")
        debug_mode = st.checkbox("Ativar modo debug")
        if password != ADMIN_PASSWORD:
            st.caption("Informe a senha para acessar.")
            return
        today = datetime.now().date().isoformat()
        st.success("Acesso liberado")
        daily = load_records_by_date(today)
        st.write("Registros do dia")
        st.dataframe(daily, use_container_width=True, hide_index=True)
        st.write("Funcionarios")
        st.dataframe(list_employees(), use_container_width=True, hide_index=True)
        csv_path = export_daily_csv(today)
        st.download_button("⬇️ Baixar CSV geral do dia", csv_path.read_bytes(), csv_path.name, "text/csv")
        summary = get_daily_summary(today)
        st.json(summary)
        local_ip = _get_local_ip()
        st.divider()
        st.markdown("**Acesso pelo celular**")
        st.code(f"http://{local_ip}:8501", language=None)
        st.caption(
            "Use este endereco no celular (mesma rede Wi-Fi). "
            "Camera pode ser bloqueada via HTTP — use iniciar_app_mobile.bat "
            "para HTTPS automatico com camera funcionando."
        )
        if debug_mode and st.session_state.get("last_detection"):
            st.session_state.show_admin_debug = True
        else:
            st.session_state.show_admin_debug = False


def render_debug_panel() -> None:
    if not st.session_state.get("show_admin_debug") or not st.session_state.get("last_detection"):
        return
    result = st.session_state.last_detection
    debug = result.get("debug_images", {})
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Debug tecnico da ultima deteccao")
    cols = st.columns(4)
    for idx, key in enumerate(["original", "mask", "enhanced", "result"]):
        img = debug.get(key) if key != "original" else result.get("original")
        if isinstance(img, np.ndarray):
            cols[idx].image(bgr_to_rgb(img), caption=key, use_column_width=True)
    st.json(result.get("metadata", {}))
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_db()
    st.set_page_config(page_title="MetaCheck AI - Hardware Counter", page_icon="*", layout="wide")
    inject_css()
    render_brand_header()
    employee_ready = employee_login()
    render_admin_area()

    if not employee_ready:
        render_capture_guide()
        st.info("Informe nome e matricula na lateral para liberar a captura e a contagem.")
        return
    if "last_detection" not in st.session_state:
        st.session_state.last_detection = None

    render_capture()
    render_result()
    render_debug_panel()
    render_employee_history()

if __name__ == "__main__":
    main()
