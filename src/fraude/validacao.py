"""Validação cruzada temporal (janela expansiva) reutilizada nos notebooks 04, 05 e 06."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as cfg
from .dados import folds_temporais
from .features import ConstrutorFeatures
from .metricas import recall_em_fpr


@dataclass
class Fold:
    numero: int
    X_tr: pd.DataFrame
    y_tr: np.ndarray
    X_va: pd.DataFrame
    y_va: np.ndarray
    idx_va: np.ndarray   # posições (na base dev) das linhas de validação


def construir_folds(dev: pd.DataFrame, **params_construtor) -> list[Fold]:
    """Para cada fold temporal, ajusta o construtor de features SÓ no passado e transforma a janela futura.

    As features dependentes do alvo do treino saem out-of-fold (ver ConstrutorFeatures.fit_transform).
    """
    dev = dev.reset_index(drop=True)
    folds = []
    for k, (idx_tr, idx_va) in enumerate(folds_temporais(dev[cfg.COL_DATA]), start=1):
        tr, va = dev.iloc[idx_tr], dev.iloc[idx_va]
        cf = ConstrutorFeatures(**params_construtor)
        X_tr = cf.fit_transform(tr, tr[cfg.ALVO])
        X_va = cf.transform(va)
        folds.append(Fold(k, X_tr.reset_index(drop=True), tr[cfg.ALVO].to_numpy(),
                          X_va.reset_index(drop=True), va[cfg.ALVO].to_numpy(), idx_va))
    return folds


def avaliar_cv(modelo, folds: list[Fold], colunas=None, nome=None):
    """Treina `modelo` em cada fold e devolve (métricas por fold, predições out-of-fold)."""
    linhas, oof = [], []
    for f in folds:
        cols = colunas or list(f.X_tr.columns)
        m = clone(modelo)
        t0 = time.perf_counter()
        m.fit(f.X_tr[cols], f.y_tr)
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = m.predict_proba(f.X_va[cols])[:, 1]
        t_pred = (time.perf_counter() - t0) / len(p) * 1e6  # µs por transação (em lote)
        linhas.append({
            "modelo": nome, "fold": f.numero,
            "PR AUC": average_precision_score(f.y_va, p),
            "ROC AUC": roc_auc_score(f.y_va, p),
            "Recall @ FPR 5%": recall_em_fpr(f.y_va, p, 0.05),
            "treino (s)": t_treino, "predição (µs/transação)": t_pred,
        })
        oof.append(pd.DataFrame({"idx": f.idx_va, "fold": f.numero, "y": f.y_va, "p": p}))
    return pd.DataFrame(linhas), pd.concat(oof, ignore_index=True)


def resumo_cv(resultados: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio-padrão das métricas por modelo."""
    metr = ["PR AUC", "ROC AUC", "Recall @ FPR 5%", "treino (s)", "predição (µs/transação)"]
    g = resultados.groupby("modelo", sort=False)[metr]
    out = g.mean().add_suffix(" (média)").join(g.std()[["PR AUC", "ROC AUC"]].add_suffix(" (dp)"))
    return out.sort_values("PR AUC (média)", ascending=False)
