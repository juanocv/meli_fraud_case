"""Configurações centrais do projeto: caminhos, recortes temporais e premissas de negócio."""
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_BRUTO = RAIZ / "dados.csv"
DIR_PROCESSADO = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"

ARQUIVO_BASE = DIR_PROCESSADO / "base.parquet"          # base limpa + coluna 'particao'
ARQUIVO_FEATURES = DIR_PROCESSADO / "features.parquet"  # matriz de features (saída do nb 04)
ARQUIVO_MODELO = DIR_MODELOS / "modelo_fraude.joblib"   # pipeline final (nb 06)
ARQUIVO_PRED_TESTE = DIR_PROCESSADO / "predicoes_teste.parquet"
ARQUIVO_PRED_OOF = DIR_PROCESSADO / "predicoes_oof_dev.parquet"

# --------------------------------------------------------------------------
# Colunas
# --------------------------------------------------------------------------
ALVO = "fraude"
COL_DATA = "fecha"
COL_VALOR = "monto"
COL_SCORE_ATUAL = "score"

# --------------------------------------------------------------------------
# Recortes temporais (out-of-time). Datas de início inclusivas.
#   treino     : 2020-03-08 -> 2020-03-31
#   validacao  : 2020-04-01 -> 2020-04-10
#   teste (OOT): 2020-04-11 -> 2020-04-21  (só é tocado na avaliação final)
# --------------------------------------------------------------------------
INICIO_VALIDACAO = pd.Timestamp("2020-04-01")
INICIO_TESTE = pd.Timestamp("2020-04-11")

# Folds de validação cruzada temporal (janela expansiva) dentro de treino+validação.
# Cada tupla = (início da janela de validação, fim exclusivo da janela).
FOLDS_TEMPORAIS = [
    (pd.Timestamp("2020-03-22"), pd.Timestamp("2020-03-29")),
    (pd.Timestamp("2020-03-29"), pd.Timestamp("2020-04-05")),
    (pd.Timestamp("2020-04-05"), pd.Timestamp("2020-04-11")),
]

# --------------------------------------------------------------------------
# Premissas de negócio
# --------------------------------------------------------------------------
MARGEM_APROVACAO = 0.10   # receita sobre pagamento legítimo aprovado (10% do valor)
PERDA_FRAUDE = 1.00       # perda sobre fraude aprovada (100% do valor)
RAZAO_CUSTO_FN_FP = 10    # pergunta 5: fraude custa 10x um bloqueio indevido

SEED = 42
