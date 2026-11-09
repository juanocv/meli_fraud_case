"""Estilo visual único para todos os notebooks + gráficos reutilizados.

Paleta categórica validada (ordem fixa, segura para daltonismo nos pares adjacentes):
    1 azul  -> novo modelo / transações legítimas
    2 laranja -> score atual / fraude
    3 verde-água -> terceira série quando necessário
Regras seguidas: um único eixo y por gráfico, linhas de 2px, grade em linha fina
sólida e discreta, legenda sempre que houver 2+ séries.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
AMARELO = "#eda100"
MAGENTA = "#e87ba4"
CINZA = "#898781"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#fcfcfb"

CATEGORICA = [AZUL, LARANJA, VERDE_AGUA, AMARELO, MAGENTA, "#008300", "#6250d6", "#e34948"]
# Rampa sequencial (azul) para mapas de calor / magnitude
SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

COR_ALVO = {0: AZUL, 1: LARANJA}
ROTULO_ALVO = {0: "Legítima", 1: "Fraude"}


def aplicar_estilo():
    """Aplica o tema padrão do projeto ao matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EIXO,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
        "grid.color": GRADE,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "text.color": TINTA,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "patch.linewidth": 0,
    })


def formatar_pct(ax, eixo="y", casas=0):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=casas)
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


def taxa_fraude_por_faixa(df, coluna, alvo="fraude", bins=10, ax=None, titulo=None,
                          cor=AZUL, rotulo_x=None):
    """Barra com a taxa de fraude por faixa (quantis) de uma variável numérica.

    Nulos viram uma faixa própria ("nulo"), pois em fraude a ausência de
    informação costuma ser informativa.
    """
    ax = ax or plt.gca()
    s = df[coluna]
    faixas = pd.qcut(s, q=bins, duplicates="drop")
    rot = faixas.astype("string").fillna("nulo")
    tab = (df.assign(_f=rot).groupby("_f", observed=True, sort=False)[alvo]
           .agg(["mean", "size"]))
    ordem = [c for c in faixas.cat.categories.astype(str)] + (["nulo"] if s.isna().any() else [])
    tab = tab.reindex([o for o in ordem if o in tab.index])
    cores = [CINZA if i == "nulo" else cor for i in tab.index]
    ax.bar(range(len(tab)), tab["mean"], color=cores, width=0.8)
    ax.axhline(df[alvo].mean(), color=TINTA_2, lw=1)
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels(tab.index, rotation=45, ha="right", fontsize=7)
    formatar_pct(ax)
    ax.set_title(titulo or f"Taxa de fraude por faixa de '{coluna}'")
    ax.set_xlabel(rotulo_x or coluna)
    ax.set_ylabel("Taxa de fraude")
    return tab


def curvas_roc_pr(y, scores: dict, axes=None, cores=None):
    """ROC e Precision-Recall lado a lado para um ou mais scores."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cores = cores or CATEGORICA
    for (nome, s), cor in zip(scores.items(), cores):
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, color=cor, label=f"{nome} (AUC={roc_auc_score(y, s):.3f})")
        p, r, _ = precision_recall_curve(y, s)
        axes[1].plot(r, p, color=cor, label=f"{nome} (AP={average_precision_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], color=EIXO, lw=1)
    axes[0].set(title="Curva ROC", xlabel="Taxa de falso positivo", ylabel="Recall (taxa de verdadeiro positivo)")
    axes[1].axhline(np.mean(y), color=EIXO, lw=1)
    axes[1].set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precisão")
    for ax in axes:
        ax.legend(loc="best")
    return axes
