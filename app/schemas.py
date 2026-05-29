from datetime import datetime

from pydantic import BaseModel


class ContagemOut(BaseModel):
    id: int
    funcionario_nome: str
    data_hora: datetime
    total_parafusos: int
    confianca_media: float
    imagem_original: str
    imagem_processada: str
    observacao: str | None = None

    class Config:
        from_attributes = True
