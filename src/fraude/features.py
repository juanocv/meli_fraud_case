"""Construção de features — mesmo código no treino, na avaliação e na inferência online.

Todas as features usam apenas informação disponível NO MOMENTO da transação
(payload da requisição + tabelas de referência aprendidas no treino).

Features que dependem do alvo (codificação da categoria `j` e score de texto
do título `i`) são geradas *out-of-fold* durante o `fit_transform`, para que o
modelo seguinte não treine com valores "contaminados" pelo próprio rótulo.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Variáveis numéricas anônimas usadas como vieram (árvores lidam com NaN nativamente).
# 'k' é deliberadamente excluída: é única por linha e não tem relação com o alvo (ver nb 03/04).
NUMERICAS_BRUTAS = ["a", "b", "c", "d", "e", "f", "h", "l", "m", "n", "monto"]

_RE_NAO_ALFANUM = re.compile(r"[^a-z0-9 ]+")


def normalizar_texto(s) -> str:
    """Minúsculas, sem acentos e só caracteres alfanuméricos."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return _RE_NAO_ALFANUM.sub(" ", s)


class ModeloTexto:
    """TF-IDF (palavras e bigramas) + regressão logística sobre o título do produto."""

    def __init__(self, C=0.5, min_df=5, max_features=50_000):
        self.C, self.min_df, self.max_features = C, min_df, max_features

    def fit(self, textos, y):
        self.vet_ = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df,
                                    max_features=self.max_features, sublinear_tf=True,
                                    dtype=np.float32)
        x = self.vet_.fit_transform(textos)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(x, y)
        return self

    def predict(self, textos):
        return self.lr_.predict_proba(self.vet_.transform(textos))[:, 1]


class ConstrutorFeatures(BaseEstimator, TransformerMixin):
    """Transforma o payload bruto da transação na matriz de features do modelo.

    Parâmetros
    ----------
    usar_categoria : inclui codificação da categoria do produto (`j`).
    usar_texto     : inclui o score de risco do título do produto (`i`).
    alpha_te       : suavização (pseudo-contagem) da codificação pelo alvo.
    n_folds_oof    : nº de folds para gerar as features dependentes do alvo no treino.
    """

    def __init__(self, usar_categoria=True, usar_texto=True, alpha_te=50, n_folds_oof=5, seed=42):
        self.usar_categoria = usar_categoria
        self.usar_texto = usar_texto
        self.alpha_te = alpha_te
        self.n_folds_oof = n_folds_oof
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        # Países com volume suficiente viram colunas próprias; o resto cai em "outros".
        cont = X["g"].value_counts()
        self.paises_ = sorted(cont[cont >= 500].index.tolist())
        if self.usar_categoria:
            self.mapa_te_j_, self.mapa_freq_j_ = self._ajustar_categoria(X["j"], y)
        if self.usar_texto:
            self.modelo_texto_ = ModeloTexto().fit(self._textos(X), y)
        self.feature_names_ = list(self.transform(X.head(2)).columns)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **_):
        """Ajusta e devolve features do treino com as colunas dependentes do alvo out-of-fold."""
        self.fit(X, y)
        out = self.transform(X)
        if self.usar_categoria or self.usar_texto:
            y = np.asarray(y)
            skf = StratifiedKFold(self.n_folds_oof, shuffle=True, random_state=self.seed)
            te_oof = np.zeros(len(X))
            txt_oof = np.zeros(len(X))
            textos = self._textos(X) if self.usar_texto else None
            for idx_a, idx_b in skf.split(X, y):
                if self.usar_categoria:
                    mapa, _ = self._ajustar_categoria(X["j"].iloc[idx_a], y[idx_a])
                    te_oof[idx_b] = X["j"].iloc[idx_b].map(mapa).astype(float).fillna(self.prior_).to_numpy()
                if self.usar_texto:
                    m = ModeloTexto().fit(textos.iloc[idx_a], y[idx_a])
                    txt_oof[idx_b] = m.predict(textos.iloc[idx_b])
            if self.usar_categoria:
                out["cat_taxa_fraude"] = te_oof
            if self.usar_texto:
                out["titulo_score"] = txt_oof
        return out

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in NUMERICAS_BRUTAS:
            out[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        # o: três estados com risco bem distinto (N >> Y >> nulo) -> duas dummies
        o = X["o"].astype("string")
        out["o_Y"] = (o == "Y").fillna(False).astype(int)
        out["o_N"] = (o == "N").fillna(False).astype(int)
        # p: binária Y/N
        out["p_Y"] = (X["p"].astype("string") == "Y").fillna(False).astype(int)

        # g (país): dummies dos países relevantes + flag de nulo
        g = X["g"].astype("string")
        for pais in self.paises_:
            out[f"pais_{pais}"] = (g == pais).fillna(False).astype(int)
        out["pais_nulo"] = g.isna().astype(int)

        # Data/hora: o risco de fraude é muito maior de madrugada
        data = pd.to_datetime(X["fecha"])
        out["hora"] = data.dt.hour
        out["madrugada"] = data.dt.hour.between(0, 5).astype(int)
        out["dia_semana"] = data.dt.day_of_week

        if self.usar_categoria:
            j = X["j"].astype("string")
            out["cat_taxa_fraude"] = j.map(self.mapa_te_j_).astype(float).fillna(self.prior_).to_numpy()
            out["cat_volume_log"] = np.log1p(j.map(self.mapa_freq_j_).astype(float).fillna(0).to_numpy())
        if self.usar_texto:
            out["titulo_score"] = self.modelo_texto_.predict(self._textos(X))
        return out

    # ------------------------------------------------------------- helpers
    def _ajustar_categoria(self, j: pd.Series, y: np.ndarray):
        """Taxa de fraude suavizada por categoria (shrinkage para a média global)."""
        tab = pd.DataFrame({"j": j.astype("string").to_numpy(), "y": y}).groupby("j")["y"].agg(["sum", "count"])
        prior = float(np.mean(y))
        taxa = (tab["sum"] + self.alpha_te * prior) / (tab["count"] + self.alpha_te)
        return taxa.to_dict(), tab["count"].to_dict()

    @staticmethod
    def _textos(X):
        return X["i"].fillna("").map(normalizar_texto)

    def get_feature_names_out(self, *_):
        return np.array(self.feature_names_)
