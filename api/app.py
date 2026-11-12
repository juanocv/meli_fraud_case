"""API de decisão antifraude (FastAPI).

Para subir localmente (na raiz do repositório):

    .venv/Scripts/python -m uvicorn api.app:app --port 8000 --no-access-log

Documentação interativa (Swagger): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from fraude import config as cfg  # noqa: E402
from fraude.inferencia import ModeloFraude  # noqa: E402

CAMINHO_MODELO = Path(os.getenv("MODELO_FRAUDE_PATH", str(cfg.ARQUIVO_MODELO)))
TAMANHO_MAX_LOTE = 1000

# Log de auditoria: uma linha JSON por decisão (sem dados pessoais do payload).
# Em produção, o stdout do container vai para o CloudWatch e, via Firehose, para o S3.
log_decisoes = logging.getLogger("api.decisoes")
if not log_decisoes.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    log_decisoes.addHandler(_h)
    log_decisoes.setLevel(logging.INFO)
    log_decisoes.propagate = False


# ---------------------------------------------------------------------------
# Contrato de entrada e saída
# ---------------------------------------------------------------------------
EXEMPLO = {
    "a": 3, "b": 0.7322, "c": 28702.11, "d": 50.0, "e": 0.5714, "f": 19.0, "g": "BR", "h": 10,
    "i": "Engate Reboque E Rabicho Golf 2008 2009 2010 2011 Ate 2014", "j": "cat_2da0eeb",
    "k": 0.2119, "l": 1400.0, "m": 1399.0, "n": 1, "o": "Y", "p": "Y",
    "fecha": "2020-04-21T15:51:11", "monto": 27.01,
}


class Transacao(BaseModel):
    """Payload de um pagamento, no mesmo formato da base de treino."""

    model_config = ConfigDict(json_schema_extra={"example": EXEMPLO})

    a: int | None = None
    b: float | None = None
    c: float | None = None
    d: float | None = None
    e: float | None = None
    f: float | None = None
    g: str | None = Field(None, description="País (código ISO de 2 letras)")
    h: int | None = None
    i: str | None = Field(None, description="Título do produto")
    j: str | None = Field(None, description="Categoria do produto")
    k: float | None = Field(None, description="Aceita, mas não usada pelo modelo")
    l: float | None = None
    m: float | None = None
    n: int | None = None
    o: Literal["Y", "N"] | None = None
    p: Literal["Y", "N"] | None = None
    fecha: datetime = Field(..., description="Data e hora da transação (horário local, sem fuso)")
    monto: float = Field(..., gt=0, description="Valor do pagamento")

    @field_validator("g", "i", "j", "o", "p", mode="before")
    @classmethod
    def vazio_vira_nulo(cls, v):
        """Texto vazio ou 'null' vira nulo, que é como a ausência aparece no treino."""
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "nan"}:
            return None
        return v


class Motivo(BaseModel):
    feature: str
    valor: float | None
    contribuicao: float = Field(..., description="Contribuição SHAP em log-odds (antes da calibração)")


class Decisao(BaseModel):
    probabilidade_fraude: float
    decisao: Literal["APROVAR", "REJEITAR"]
    limiar: float
    versao_modelo: str | None
    motivos: list[Motivo] | None = None


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------
@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega o artefato uma única vez, na inicialização (nunca por requisição)."""
    modelo = ModeloFraude.carregar(CAMINHO_MODELO)
    if modelo.limiar is None:
        raise RuntimeError("Artefato sem limiar de decisão: execute o notebook 08 antes de subir a API.")
    app.state.modelo = modelo
    app.state.carregado_em = datetime.now().isoformat(timespec="seconds")
    yield


class TempoServidor:
    """Middleware ASGI leve: informa no header quanto tempo o servidor levou para responder."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.perf_counter()

        async def enviar(mensagem):
            if mensagem["type"] == "http.response.start":
                ms = (time.perf_counter() - t0) * 1e3
                mensagem.setdefault("headers", []).append((b"x-tempo-servidor-ms", f"{ms:.3f}".encode()))
            await send(mensagem)

        await self.app(scope, receive, enviar)


app = FastAPI(
    title="API de decisão antifraude",
    version="1.0.0",
    description="Avalia o risco de fraude de um pagamento e decide entre aprovar e rejeitar.",
    lifespan=ciclo_de_vida,
)
app.add_middleware(TempoServidor)


@app.get("/saude", summary="Health check (usado pelo balanceador de carga)")
def saude():
    m: ModeloFraude = app.state.modelo
    return {
        "status": "ok",
        "versao_modelo": m.metadados.get("versao"),
        "limiar": m.limiar,
        "n_features": len(m.construtor.feature_names_),
        "carregado_em": app.state.carregado_em,
    }


@app.post("/v1/fraude/avaliar", response_model=Decisao, response_model_exclude_none=True,
          summary="Avalia uma transação")
def avaliar(transacao: Transacao, response: Response,
            explicar: bool = Query(False, description="Inclui os 3 principais motivos do risco")):
    m: ModeloFraude = app.state.modelo
    t0 = time.perf_counter()
    resultado = m.pontuar(transacao.model_dump(), explicar=explicar)
    ms = (time.perf_counter() - t0) * 1e3
    response.headers["x-tempo-modelo-ms"] = f"{ms:.3f}"
    log_decisoes.info(json.dumps({
        "evento": "decisao", "versao": resultado["versao_modelo"], "decisao": resultado["decisao"],
        "probabilidade": resultado["probabilidade_fraude"], "tempo_modelo_ms": round(ms, 3),
    }))
    return {**resultado, "limiar": m.limiar}


@app.post("/v1/fraude/avaliar-lote", response_model=list[Decisao], response_model_exclude_none=True,
          summary=f"Avalia um lote de até {TAMANHO_MAX_LOTE} transações (reprocessamentos e backtests)")
def avaliar_lote(transacoes: Annotated[list[Transacao], Body(max_length=TAMANHO_MAX_LOTE)]):
    if not transacoes:
        raise HTTPException(status_code=422, detail="Lote vazio.")
    m: ModeloFraude = app.state.modelo
    df = pd.DataFrame([t.model_dump() for t in transacoes])
    probs = m.probabilidade(df)
    versao = m.metadados.get("versao")
    return [{"probabilidade_fraude": round(float(p), 6),
             "decisao": "REJEITAR" if p >= m.limiar else "APROVAR",
             "limiar": m.limiar, "versao_modelo": versao} for p in probs]
