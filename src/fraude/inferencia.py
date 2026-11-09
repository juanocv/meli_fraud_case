"""Artefato de produção: features + modelo + calibração + política de decisão.

Um único objeto serializado garante que o serviço online use exatamente o mesmo
código de features do treino (evita *training-serving skew*).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .features import NUMERICAS_BRUTAS, ConstrutorFeatures, normalizar_texto


class CalibradorPlatt:
    """Calibração de Platt sobre o logit do score: p_cal = sigmoid(a * logit(p) + b).

    Com só 2 parâmetros e estritamente monotônica, preserva a ordenação (AUC/PR AUC idênticos)
    e corrige o deslocamento de taxa base (prior shift) e a inclinação das probabilidades.
    """

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        self.lr_ = LogisticRegression(C=1e6).fit(self._logit(p).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, p):
        return self.lr_.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


@dataclass
class ModeloFraude:
    construtor: ConstrutorFeatures
    modelo: object                      # LGBMClassifier já treinado
    calibrador: object | None = None    # IsotonicRegression (score bruto -> probabilidade)
    limiar: float | None = None         # rejeita se probabilidade >= limiar
    metadados: dict = field(default_factory=dict)

    # ----------------------------------------------------------- lote (batch)
    def score_bruto(self, df: pd.DataFrame) -> np.ndarray:
        X = self.construtor.transform(df)[self.construtor.feature_names_]
        return self.modelo.predict_proba(X)[:, 1]

    def probabilidade(self, df: pd.DataFrame) -> np.ndarray:
        s = self.score_bruto(df)
        return self.calibrador.predict(s) if self.calibrador is not None else s

    def decidir(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(self.probabilidade(df) >= self.limiar, "REJEITAR", "APROVAR")

    # -------------------------------------------- online (uma transação/dict)
    def vetor_features(self, t: dict) -> np.ndarray:
        """Caminho rápido: monta o vetor de features direto do payload, sem pandas.

        Replica `ConstrutorFeatures.transform` linha a linha; a paridade entre os
        dois caminhos é testada no notebook 09.
        """
        c = self.construtor
        v = []
        for col in NUMERICAS_BRUTAS:
            x = t.get(col)
            v.append(np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x))
        o, p, g = t.get("o"), t.get("p"), t.get("g")
        v += [float(o == "Y"), float(o == "N"), float(p == "Y")]
        v += [float(g == pais) for pais in c.paises_]
        v.append(float(g is None or (isinstance(g, float) and np.isnan(g))))
        data = pd.Timestamp(t["fecha"])
        v += [data.hour, float(0 <= data.hour <= 5), data.dayofweek]
        if c.usar_categoria:
            v.append(c.mapa_te_j_.get(t.get("j"), c.prior_))
            v.append(np.log1p(c.mapa_freq_j_.get(t.get("j"), 0)))
        if c.usar_texto:
            v.append(c.modelo_texto_.predict([normalizar_texto(t.get("i") or "")])[0])
        return np.asarray(v, dtype=float).reshape(1, -1)

    def pontuar(self, t: dict) -> dict:
        """Resposta da API para uma transação."""
        # num_threads=1: para uma única linha, paralelizar só adiciona overhead
        s = self.modelo.booster_.predict(self.vetor_features(t), num_threads=1)[0]
        p = float(self.calibrador.predict([s])[0]) if self.calibrador is not None else float(s)
        return {
            "probabilidade_fraude": round(p, 6),
            "decisao": "REJEITAR" if p >= self.limiar else "APROVAR",
            "versao_modelo": self.metadados.get("versao"),
        }

    # ------------------------------------------------------- persistência
    def salvar(self, caminho):
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho) -> "ModeloFraude":
        return joblib.load(caminho)
