from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from app.database import Base


class Contagem(Base):
    __tablename__ = "contagens"

    id = Column(Integer, primary_key=True, index=True)
    funcionario_nome = Column(String(160), nullable=False)
    data_hora = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    total_parafusos = Column(Integer, nullable=False)
    confianca_media = Column(Float, nullable=False)
    imagem_original = Column(Text, nullable=False)
    imagem_processada = Column(Text, nullable=False)
    observacao = Column(Text, nullable=True)
