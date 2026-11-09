"""Leitura do dado bruto, tipagem e particionamento temporal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def carregar_bruto(caminho=cfg.ARQUIVO_BRUTO) -> pd.DataFrame:
    """Lê o CSV entregue pelo time de Analytics com a tipagem correta."""
    df = pd.read_csv(caminho, parse_dates=[cfg.COL_DATA])
    # Categóricas de texto: mantemos NaN explícito (o nulo é informativo, ver nb 01)
    for col in ["g", "i", "j", "o", "p"]:
        df[col] = df[col].astype("string")
    return df


def atribuir_particao(df: pd.DataFrame) -> pd.Series:
    """Rotula cada transação como treino / validacao / teste pela data (out-of-time)."""
    data = df[cfg.COL_DATA]
    return pd.Series(
        np.select(
            [data < cfg.INICIO_VALIDACAO, data < cfg.INICIO_TESTE],
            ["treino", "validacao"],
            default="teste",
        ),
        index=df.index,
        name="particao",
    )


def carregar_base() -> pd.DataFrame:
    """Lê a base já limpa e particionada (gerada no notebook 01)."""
    return pd.read_parquet(cfg.ARQUIVO_BASE)


def folds_temporais(datas: pd.Series, folds=cfg.FOLDS_TEMPORAIS):
    """Gera (idx_treino, idx_valid) em janela expansiva.

    Para cada janela de validação [inicio, fim), o treino usa TUDO que é
    anterior a `inicio`. Assim nenhuma informação do futuro vaza para o passado,
    simulando o que acontece em produção (treina-se com o passado, prediz-se o futuro).
    """
    datas = pd.Series(datas).reset_index(drop=True)
    for inicio, fim in folds:
        idx_tr = np.where(datas < inicio)[0]
        idx_va = np.where((datas >= inicio) & (datas < fim))[0]
        yield idx_tr, idx_va
