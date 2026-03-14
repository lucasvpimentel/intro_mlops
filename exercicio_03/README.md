# Exercicio 03 — Classificador de Qualidade de Vinhos

Classificacao de vinhos em 3 cultivares usando Random Forest. Inclui analise
exploratoria em notebook, pipeline automatizado via `.bat` e inferencia em lote
(batch inference) com entrada JSON e saida CSV.

## Estrutura do projeto

```
exercicio_03/
├── data/
│   ├── raw.csv                  # Dataset bruto
│   ├── processed.csv            # Dataset normalizado
│   ├── scaler.joblib            # StandardScaler ajustado no treino
│   ├── wine_model.joblib        # Modelo treinado
│   ├── evaluation.txt           # Relatorio de avaliacao
│   ├── input.json               # Entrada para inferencia em lote (10 vinhos)
│   └── output.csv               # Saida da inferencia em lote
├── notebooks/
│   └── analise_correlacao.ipynb # EDA e analise de correlacao
├── scripts/
│   └── run_pipeline.bat         # Pipeline completo via terminal Windows
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_data.py     # Baixa e salva o dataset
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # Normaliza features e salva o scaler
│   └── models/
│       ├── __init__.py
│       ├── train.py             # Treina e serializa o modelo
│       ├── evaluate.py          # Avalia e gera evaluation.txt
│       └── predict.py           # Inferencia em lote: input.json -> output.csv
├── main.py                      # Ponto de entrada unico
└── requirements.txt
```

---

## Plano de construcao manual (passo a passo)

### Passo 1 — Criar o ambiente virtual

```bash
python -m venv venv
```

Ativar:
- **Windows:** `venv\Scripts\activate`
- **Linux/macOS:** `source venv/bin/activate`

### Passo 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

| Pacote | Finalidade |
|---|---|
| `scikit-learn` | Dataset, Random Forest, metricas |
| `pandas` | Manipulacao de CSV e JSON |
| `joblib` | Serializar scaler e modelo |
| `numpy` | Operacoes numericas |
| `matplotlib` + `seaborn` | Graficos no notebook |
| `notebook` | Executar o Jupyter Notebook |

### Passo 3 — Executar o pipeline completo

**Via main.py:**
```bash
python main.py pipeline
```

**Via script bat (Windows):**
```bat
scripts\run_pipeline.bat
```

Ambos executam as 4 etapas em sequencia:

| Etapa | O que faz |
|---|---|
| download | Carrega Wine Dataset via sklearn, salva `raw.csv` |
| features | Ajusta `StandardScaler`, salva `scaler.joblib` e `processed.csv` |
| train | Treina Random Forest (CV 5-fold), salva `wine_model.joblib` |
| evaluate | Avalia no conjunto de teste, salva `evaluation.txt` |

Saida esperada do treino:
```
CV Accuracy (5-fold): 0.9862 +/- 0.0276
Accuracy no teste:    1.0000
```

### Passo 4 — Inferencia em lote

Edite `data/input.json` com os dados dos vinhos a classificar (qualquer
quantidade de registros, cada um com as 13 features) e execute:

```bash
python main.py predict
```

O script:
1. Le `data/input.json`
2. Aplica o mesmo `scaler.joblib` do treino
3. Classifica cada vinho
4. Salva `data/output.csv` com colunas `cultivar_previsto` e `confianca_pct`

Exemplo de saida:
```
cultivar_previsto  confianca_pct
          class_0          100.0
          class_0           98.0
          class_1           99.0
          class_2           92.5
```

### Passo 5 — Analise exploratoria (notebook)

```bash
jupyter notebook notebooks/analise_correlacao.ipynb
```

O notebook cobre:
1. Visao geral do dataset e estatisticas descritivas
2. Distribuicao das 3 classes de cultivo
3. Matriz de correlacao entre as 13 features (heatmap)
4. Features mais correlacionadas com o cultivar
5. Distribuicao das top features por classe (KDE plots)
6. Pares de features com multicolinearidade alta (|r| > 0.7)

---

## Artefatos Salvos e Uso Independente

Apos rodar `pipeline` uma vez, dois artefatos ficam persistidos em disco:

```
data/
├── scaler.joblib       # StandardScaler ajustado nos dados de treino
└── wine_model.joblib   # RandomForestClassifier treinado
```

**Voce nao precisa retreinar para fazer predicoes em lote.** Com o `venv` ja
criado e os artefatos ja gerados, basta ativar, editar o `input.json` e rodar:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. (opcional) Editar data/input.json com os vinhos que deseja classificar

# 3. Prever em lote — so carrega scaler.joblib e wine_model.joblib
python main.py predict
```

O `predict` executa apenas `src/models/predict.py`, que faz:
1. Le `data/input.json` (seus dados de entrada)
2. `joblib.load("data/scaler.joblib")` — normaliza com a escala do treino
3. `joblib.load("data/wine_model.joblib")` — classifica cada vinho
4. Salva `data/output.csv` com `cultivar_previsto` e `confianca_pct`

**Formato do `input.json`** (cada vinho e um objeto com as 13 features):
```json
[
  {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
    "alcalinity_of_ash": 15.6, "magnesium": 127.0,
    "total_phenols": 2.80, "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
    "color_intensity": 5.64, "hue": 1.04,
    "od280_od315": 3.92, "proline": 1065.0
  }
]
```

Se voce deletar `data/raw.csv` e `data/processed.csv`, o `predict` ainda
funciona — ele so depende dos dois `.joblib` e do `input.json`.

---

## Como funciona internamente

```
main.py / run_pipeline.bat
  |-- download  --> src/data/download_data.py    --> data/raw.csv
  |-- features  --> src/features/build_features  --> data/scaler.joblib
  |                                                  data/processed.csv
  |-- train     --> src/models/train.py          --> data/wine_model.joblib
  |-- evaluate  --> src/models/evaluate.py       --> data/evaluation.txt
  |-- predict   --> src/models/predict.py        --> data/output.csv
  |                 (le input.json, aplica scaler, classifica em lote)
  |-- pipeline  --> download + features + train + evaluate
```

## Decisoes de projeto

- **Random Forest:** robusto a multicolinearidade entre features quimicas,
  nao requer features independentes como modelos lineares.
- **Avaliacao separada do treino (`evaluate.py`):** permite re-avaliar sem
  re-treinar, e e o passo que o `.bat` chama explicitamente.
- **Batch inference via JSON:** mais realista que argumentos CLI para cenarios
  com multiplas amostras; facilita integracao com sistemas externos.
- **`evaluation.txt` persistido:** permite auditar metricas sem re-executar
  o pipeline.
- **Notebook independente:** consome apenas `raw.csv`, sem depender dos
  artefatos de treino — pode ser rodado a qualquer momento.
