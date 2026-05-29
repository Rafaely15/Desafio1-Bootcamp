from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from app.database import Base


class Contagem(Base):
    __tablename__ = "contagens"

    id = Column(Integer, primary_key=True, index=True)
    funcionario_nome = Column(String(160), nullable=False)
    funcionario_id = Column(String(80), nullable=True)
    setor = Column(String(120), nullable=True)
    pedido = Column(String(120), nullable=True)
    data_hora = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    total_parafusos = Column(Integer, nullable=False)
    total_corrigido = Column(Integer, nullable=True)
    confianca_media = Column(Float, nullable=False)
    imagem_original = Column(Text, nullable=False)
    imagem_processada = Column(Text, nullable=False)
    status = Column(String(40), nullable=True)
    observacao = Column(Text, nullable=True)
