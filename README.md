# Meli: Prevenção a Fraudes - Case

Desenvolvimento de um novo modelo de fraude para pagamentos de e-commerce, avaliado contra o score em
produção, com definição de política de decisão orientada a lucro e um desenho de produtização.

**Respostas às perguntas do case:** [RESPOSTAS.md](RESPOSTAS.md)

## Resultado principal

Avaliação no **teste fora do tempo** (11/04–21/04/2020), período nunca usado em treino ou ajuste:

| | Score atual | **Novo modelo** |
|---|---|---|
| ROC AUC | 0,726 | **0,851** |
| PR AUC | 0,227 | **0,368** (+62%) |
| Recall em R$ @ 5% de bons bloqueados | 44% | **63%** |
| Lucro no teste (corte escolhido no passado) | R$ 79,1 mil | **R$ 95,8 mil (+21%)** |
| Latência por transação (API local, p50) | — | **~0,9 ms no modelo, ~2–3 ms na chamada HTTP** |

- **Modelo:** LightGBM (25 features, tunado com Optuna em CV temporal) + calibração de Platt.
- **Corte ótimo do score atual (pergunta 6):** aprovar score ≤ 75 e rejeitar ≥ 76. O lucro passa de
  R$ 51 mil para R$ 265 mil na base (5,2x).
- **Principal achado sobre o score atual:** uma "faixa ignorada" (score 0–25) com ~21% das fraudes, em que o
  score é praticamente aleatório. O novo modelo recupera parte delas.

## Metodologia

O projeto segue o **CRISP-DM** adaptado a fraude, com dois princípios transversais:

1. **Tudo é validado fora do tempo** (*out-of-time*);
2. **A decisão final é medida em R$**.

| # | Notebook | Etapa CRISP-DM | Pergunta |
|---|---|---|---|
| 01 | [Entendimento dos dados](notebooks/01_entendimento_dados.ipynb) | Negócio + dados | Qualidade, alvo, regimes temporais e particionamento out-of-time |
| 02 | [Diagnóstico do score atual](notebooks/02_baseline_score_atual.ipynb) | Dados | Quão bom é o modelo em produção e onde ele falha ("zona cega") |
| 03 | [Análise exploratória](notebooks/03_eda.ipynb) | Dados | Sinais de risco: histórico, horário, produto, país |
| 04 | [Engenharia de features](notebooks/04_feature_engineering.ipynb) | Preparação | Features sem vazamento + teste de ablação |
| 05 | [Seleção de algoritmos](notebooks/05_selecao_algoritmos.ipynb) | Modelagem | Logística, Random Forest, XGBoost, LightGBM, CatBoost |
| 06 | [Modelo final](notebooks/06_modelo_final.ipynb) | Modelagem | Tuning (Optuna), SHAP, calibração, artefato |
| 07 | [Avaliação fora do tempo](notebooks/07_avaliacao_oot.ipynb) | Avaliação | Novo modelo × score atual no teste |
| 08 | [Decisão de negócio](notebooks/08_decisao_negocio.ipynb) | Avaliação | Corte ótimo (lucro) e custo assimétrico 10:1 |
| 09 | [API local e caminho para produção](notebooks/09_api_producao.ipynb) | Implantação | API FastAPI: paridade com o laboratório, latência, vazão e contrato via HTTP; arquitetura AWS |

### Particionamento

| Partição | Período | Transações | Taxa de fraude | Uso |
|---|---|---|---|---|
| Treino | 08/03 – 31/03/2020 | 76.961 | 5,0% | ajuste |
| Validação | 01/04 – 10/04/2020 | 31.505 | 5,6% | tuning, calibração, corte |
| **Teste (OOT)** | 11/04 – 21/04/2020 | 41.534 | 4,5% | **avaliação final, usada uma única vez** |

A seleção de modelos usa **validação cruzada temporal em janela expansiva** (3 folds semanais)
dentro de treino + validação.

## Estrutura do repositório

```
├── dados.csv                  # base recebida do time de Analytics (não versionada)
├── notebooks/                 # 01 → 09, sequenciais e executados
├── api/                       # serviço FastAPI (app.py) usado no nb 09
├── src/fraude/                # código compartilhado entre treino, avaliação e inferência
│   ├── config.py              # caminhos, recortes temporais, premissas de negócio
│   ├── dados.py               # leitura, particionamento, folds temporais
│   ├── features.py            # ConstrutorFeatures (único código de features)
│   ├── validacao.py           # CV temporal reutilizável
│   ├── metricas.py            # métricas estatísticas, lucro, custo, PSI, bootstrap
│   ├── inferencia.py          # ModeloFraude: artefato de produção + caminho online rápido
│   └── graficos.py            # padrão visual dos notebooks
├── data/processed/            # saídas intermediárias (geradas pelos notebooks)
├── models/                    # artefato final modelo_fraude.joblib (gerado no nb 06/08)
├── RESPOSTAS.md               # respostas às 8 perguntas
└── requirements.txt
```

## Como reproduzir

Requer Python 3.13. Coloque o `dados.csv` na raiz do repositório.

### Windows

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

### Linux / macOS

O ambiente virtual guarda os executáveis em `bin/` em vez de `Scripts/`. Se a distribuição não
tiver o Python 3.13 (o Ubuntu 22.04, por exemplo, traz o 3.10), a forma mais simples de obtê-lo sem
privilégios de administrador é pelo [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv python install 3.13 && uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Com um Python 3.13 já instalado, o caminho sem o uv é o mesmo do Windows, trocando `Scripts` por
`bin`:

```bash
python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

### Execução

Os notebooks devem ser executados **em ordem** (cada um consome saídas do anterior). Para executar
todos de uma vez (Windows):

```bash
cd notebooks && for nb in 0*.ipynb; do ../.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Em Linux / macOS:

```bash
cd notebooks && for nb in 0*.ipynb; do ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Tempo total aproximado: 10 a 15 minutos em um notebook comum.

### API local

Depois dos notebooks 06 e 08, que geram o artefato do modelo e o limiar de decisão, a API sobe a
partir da raiz do repositório (em Linux / macOS, troque `Scripts` por `bin`):

```bash
.venv/Scripts/python -m uvicorn api.app:app --port 8000 --no-access-log
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`. Para avaliar uma transação pelo
terminal:

```bash
curl -X POST "http://127.0.0.1:8000/v1/fraude/avaliar?explicar=true" -H "Content-Type: application/json" -d '{"a": 3, "b": 0.73, "c": 28702.1, "d": 50, "e": 0.57, "f": 19, "g": "BR", "h": 10, "i": "Samsung Galaxy A10 32 Gb", "j": "cat_43b9c10", "l": 1400, "m": 1399, "n": 1, "o": "N", "p": "N", "fecha": "2020-04-21T02:51:11", "monto": 899.0}'
```

O notebook 09 sobe a própria instância da API (porta 8001) e a encerra no fim.

<!-- 
## Limitações e próximos passos

- **Variáveis anonimizadas:** a interpretação de negócio de `a`–`p` é inferida pelo comportamento.
  Com o dicionário real, várias hipóteses de features ficariam mais precisas.
- **Sem identificador de usuário, cartão ou dispositivo:** as features mais poderosas em fraude
  (velocidade e histórico por entidade, grafos de relacionamento) não puderam ser construídas. É a
  principal frente da v2, junto com um *feature store online*.
- **Base amostral e de 45 dias, no início da pandemia:** confirmar o desenho amostral e a maturação dos
  rótulos antes de calibrar a política em produção. O período tem regimes atípicos (nb 01).
- **Score atual traz informação complementar** (o *stacking* melhora o topo do ranking, nb 07):
  investigar quais insumos ele usa e incorporá-los como features próprias.
