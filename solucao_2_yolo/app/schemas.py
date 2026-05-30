from datetime import datetime

from pydantic import BaseModel


class ContagemOut(BaseModel):
    id: int
    funcionario_nome: str
    funcionario_id: str | None = None
    setor: str | None = None
    pedido: str | None = None
    data_hora: datetime
    total_parafusos: int
    total_corrigido: int | None = None
    confianca_media: float
    imagem_original: str
    imagem_processada: str
    status: str | None = None
    observacao: str | None = None

    class Config:
        from_attributes = True
