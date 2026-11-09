"""Métricas estatísticas e de negócio usadas para avaliar scores de fraude.

Convenção: score maior = maior risco. Para um ponto de corte `c`,
APROVA-SE quando score < c e REJEITA-SE quando score >= c.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from . import config as cfg


# ---------------------------------------------------------------------------
# Métricas de ordenação (independentes de ponto de corte)
# ---------------------------------------------------------------------------
def ks(y, s) -> float:
    """Estatística KS: distância máxima entre as distribuições do score em fraude e legítima."""
    y, s = np.asarray(y), np.asarray(s)
    return ks_2samp(s[y == 1], s[y == 0]).statistic


def recall_em_fpr(y, s, fpr_alvo: float) -> float:
    """Recall (fraudes capturadas) quando se aceita bloquear no máximo `fpr_alvo` dos legítimos."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_alvo, fpr, tpr))


def recall_valor_em_fpr(y, s, valor, fpr_alvo: float) -> float:
    """Mesma ideia do recall@FPR, mas em valor (R$) de fraude capturado."""
    y, s, valor = map(np.asarray, (y, s, valor))
    limiar = np.quantile(s[y == 0], 1 - fpr_alvo)
    bloq = s >= limiar
    return float(valor[bloq & (y == 1)].sum() / valor[y == 1].sum())


def resumo(y, s, valor=None, nome=None) -> pd.Series:
    """Painel padrão de métricas de um score."""
    out = {
        "ROC AUC": roc_auc_score(y, s),
        "PR AUC (AP)": average_precision_score(y, s),
        "KS": ks(y, s),
        "Recall @ FPR 1%": recall_em_fpr(y, s, 0.01),
        "Recall @ FPR 5%": recall_em_fpr(y, s, 0.05),
        "Recall @ FPR 10%": recall_em_fpr(y, s, 0.10),
    }
    if valor is not None:
        out["Recall R$ @ FPR 5%"] = recall_valor_em_fpr(y, s, valor, 0.05)
    return pd.Series(out, name=nome)


# ---------------------------------------------------------------------------
# Métricas de negócio
# ---------------------------------------------------------------------------
def resultado_transacao(y, valor, margem=cfg.MARGEM_APROVACAO, perda=cfg.PERDA_FRAUDE):
    """Resultado financeiro de APROVAR cada transação (rejeitar = 0)."""
    y, valor = np.asarray(y), np.asarray(valor, dtype=float)
    return np.where(y == 1, -perda * valor, margem * valor)


def curva_lucro(y, s, valor, cortes=None, margem=cfg.MARGEM_APROVACAO,
                perda=cfg.PERDA_FRAUDE) -> pd.DataFrame:
    """Lucro e indicadores operacionais para cada ponto de corte (aprova se s < corte)."""
    y, s, valor = np.asarray(y), np.asarray(s, dtype=float), np.asarray(valor, dtype=float)
    if cortes is None:
        cortes = np.unique(np.concatenate([np.unique(s), [s.max() + 1e-9]]))
    r = resultado_transacao(y, valor, margem, perda)
    ordem = np.argsort(s)
    s_ord, r_ord, y_ord, v_ord = s[ordem], r[ordem], y[ordem], valor[ordem]
    lucro_acum = np.concatenate([[0], np.cumsum(r_ord)])
    fraude_acum = np.concatenate([[0], np.cumsum(y_ord)])
    vfraude_acum = np.concatenate([[0], np.cumsum(v_ord * y_ord)])
    n_aprov = np.searchsorted(s_ord, cortes, side="left")  # qtd com s < corte
    tot_fraude, tot_vfraude, n = y.sum(), (valor * y).sum(), len(y)
    legit_aprov = n_aprov - fraude_acum[n_aprov]
    return pd.DataFrame({
        "corte": cortes,
        "lucro": lucro_acum[n_aprov],
        "taxa_aprovacao": n_aprov / n,
        "fraudes_bloqueadas_pct": 1 - fraude_acum[n_aprov] / tot_fraude,
        "valor_fraude_bloqueado_pct": 1 - vfraude_acum[n_aprov] / tot_vfraude,
        "legitimas_bloqueadas_pct": 1 - legit_aprov / (n - tot_fraude),
    })


def custo_esperado(y, bloquear, custo_fn=cfg.RAZAO_CUSTO_FN_FP, custo_fp=1.0) -> float:
    """Custo total com fraude aprovada (FN) custando `custo_fn` e bloqueio indevido (FP) custando `custo_fp`."""
    y, bloquear = np.asarray(y), np.asarray(bloquear).astype(bool)
    fn = ((y == 1) & ~bloquear).sum()
    fp = ((y == 0) & bloquear).sum()
    return float(custo_fn * fn + custo_fp * fp)


# ---------------------------------------------------------------------------
# Estabilidade e incerteza
# ---------------------------------------------------------------------------
def psi(esperado, atual, bins=10) -> float:
    """Population Stability Index com faixas definidas por quantis da distribuição de referência."""
    esperado, atual = np.asarray(esperado, dtype=float), np.asarray(atual, dtype=float)
    esperado, atual = esperado[~np.isnan(esperado)], atual[~np.isnan(atual)]
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, cortes)[0] / len(esperado)
    a = np.histogram(atual, cortes)[0] / len(atual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bootstrap_diferenca(y, s_novo, s_ref, metrica=roc_auc_score, n=500, seed=cfg.SEED):
    """Intervalo de confiança (bootstrap pareado) para metrica(novo) - metrica(referência)."""
    y, s_novo, s_ref = map(np.asarray, (y, s_novo, s_ref))
    rng = np.random.default_rng(seed)
    difs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        difs.append(metrica(y[idx], s_novo[idx]) - metrica(y[idx], s_ref[idx]))
    difs = np.array(difs)
    return pd.Series({
        "diferenca": metrica(y, s_novo) - metrica(y, s_ref),
        "ic95_inf": np.percentile(difs, 2.5),
        "ic95_sup": np.percentile(difs, 97.5),
        "p_valor (dif<=0)": float((difs <= 0).mean()),
    })
