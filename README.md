# Prevenção a Fraudes: novo modelo de risco de pagamentos (FREEDOM)

Desenvolvimento de um novo modelo de fraude para pagamentos de e-commerce, avaliado contra o score em
produção, com definição de política de decisão orientada a lucro e um desenho de produtização.

**Respostas às perguntas do case:** [RESPOSTAS.md](RESPOSTAS.md)

## Resultado em uma tabela

Avaliação no **teste fora do tempo** (11/04–21/04/2020), período nunca usado em treino ou ajuste:

| | Score atual | **Novo modelo** |
|---|---|---|
| ROC AUC | 0,726 | **0,851** |
| PR AUC | 0,227 | **0,368** (+62%) |
| Recall em R$ @ 5% de bons bloqueados | 44% | **63%** |
| Lucro no teste (corte escolhido no passado) | R$ 79,1 mil | **R$ 95,8 mil (+21%)** |
| Latência por transação | — | **~0,8 ms** |

- **Modelo:** LightGBM (25 features, tunado com Optuna em CV temporal) + calibração de Platt.
- **Corte ótimo do score atual (pergunta 6):** aprovar score ≤ 75 e rejeitar ≥ 76. O lucro passa de
  R$ 51 mil para R$ 265 mil na base (5,2x).
- **Principal achado sobre o score atual:** uma "zona cega" (score 0–25) com ~21% das fraudes, em que o
  score é praticamente aleatório. O novo modelo recupera parte delas.

## Metodologia

O projeto segue o **CRISP-DM** adaptado a fraude, com dois princípios transversais:

1. **Tudo é validado fora do tempo** (*out-of-time*). Fraude é adversarial e não estacionária, e
   validação aleatória superestima o desempenho.
2. **A decisão final é medida em R$**, e não só em métricas estatísticas.

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
| 09 | [Latência e produção](notebooks/09_latencia_producao.ipynb) | Implantação | Paridade lote × online, latência, trade-offs |

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

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Os notebooks devem ser executados **em ordem** (cada um consome saídas do anterior). Para executar
todos de uma vez:

```bash
cd notebooks && for nb in 0*.ipynb; do ../.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Tempo total aproximado: 10 a 15 minutos em um notebook comum (o tuning do nb 06 leva ~3 minutos).

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
