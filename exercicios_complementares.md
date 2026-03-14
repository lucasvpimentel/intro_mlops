# Exercicios Complementares

Exercicios adicionais de MLOps usando toy datasets publicos (seaborn e sklearn).
Cada um propoe uma estrutura de pastas diferente para mostrar que nao existe
uma unica forma correta de organizar um projeto de ML.

Nenhum exercicio envolve deteccao de drift — o foco e na persistencia correta
de modelos, transformadores e no ciclo treino → inferencia.

---

## Exercicio A — Sobreviventes do Titanic

**Dataset:** `seaborn.load_dataset("titanic")`
**Tarefa:** Classificacao binaria — prever se um passageiro sobreviveu (survived = 0 ou 1)
**Features sugeridas:** pclass, sex, age, sibsp, parch, fare, embarked
**Atenção:** o dataset tem valores ausentes em `age` e `embarked` — o preprocessamento e obrigatorio

**Como carregar:**

```python
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
df.to_csv("data/raw.csv", index=False)

# Colunas disponiveis:
# survived, pclass, sex, age, sibsp, parch, fare, embarked,
# class, who, adult_male, deck, embark_town, alive, alone
print(df.shape)        # (891, 15)
print(df.dtypes)
print(df.isnull().sum())  # age: 177 nulos, deck: 688 nulos, embarked: 2 nulos
```

**O que praticar:**
- Imputacao de valores ausentes (SimpleImputer)
- Codificacao de variaveis categoricas (LabelEncoder ou pd.get_dummies)
- Classificacao binaria com probabilidade de sobrevivencia
- Salvar e carregar modelo + todos os transformadores

**Estrutura proposta — flat (tudo no nivel raiz):**

```
titanic_mlops/
├── download.py        # baixa e salva o dataset
├── prepare.py         # limpeza, imputacao, encoding, salva transformadores
├── train.py           # treina e salva o modelo
├── evaluate.py        # metricas no conjunto de teste
├── predict.py         # carrega transformadores + modelo, faz predicao
├── main.py            # orquestrador CLI
├── data/
│   ├── raw.csv
│   ├── processed.csv
│   └── models/
│       ├── model.joblib
│       ├── imputer.joblib
│       └── encoder.joblib
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --pclass 3 --sex male --age 22 --fare 7.25 --embarked S

Passageiro analisado:
  Classe     : 3ª
  Sexo       : masculino
  Idade      : 22 anos

Resultado:
  Sobreviveu  : Nao
  Probabilidade de sobrevivencia: 12.4%
```

---

## Exercicio B — Gorjetas em Restaurante

**Dataset:** `seaborn.load_dataset("tips")`
**Tarefa:** Regressao — prever o valor da gorjeta (`tip`) com base nas caracteristicas da refeicao
**Features sugeridas:** total_bill, sex, smoker, day, time, size
**Target:** tip (valor em dolares)

**Como carregar:**

```python
import seaborn as sns

df = sns.load_dataset("tips")
df.to_csv("data/tips.csv", index=False)

# Colunas disponiveis:
# total_bill (float), tip (float), sex (Female/Male),
# smoker (Yes/No), day (Sun/Sat/Thur/Fri),
# time (Dinner/Lunch), size (int 1-6)
print(df.shape)        # (244, 7)
print(df.isnull().sum())  # nenhum valor ausente
print(df["tip"].describe())
```

**O que praticar:**
- Regressao sobre um dataset pequeno e interpretavel
- Lidar com features categoricas ordinais e nominais juntas
- Salvar o scaler e os encoders separadamente
- Comparar dois algoritmos (ex: Ridge vs Random Forest)

**Estrutura proposta — src/ plano (sem subdiretorios em src/):**

```
tips_mlops/
├── src/
│   ├── __init__.py
│   ├── loader.py       # carrega o dataset do seaborn
│   ├── features.py     # encoding + scaling, salva transformadores
│   ├── trainer.py      # treina e avalia o modelo
│   └── predictor.py    # inferencia com transformadores salvos
├── data/
│   ├── tips.csv
│   ├── processed.csv
│   └── artifacts/      # model.joblib, scaler.joblib, encoders/
├── main.py
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --total-bill 24.50 --sex Female --smoker No --day Sat --time Dinner --size 3

Gorjeta estimada: R$ 3.87
(Media do dataset: R$ 2.99 | Maximo: R$ 10.00)
```

---

## Exercicio C — Preco de Diamantes

**Dataset:** `seaborn.load_dataset("diamonds")`
**Tarefa:** Regressao — prever o preco (`price`) de um diamante
**Features sugeridas:** carat, cut, color, clarity, depth, table, x, y, z
**Atenção:** cut, color e clarity sao categoricas ordinais — a ordem importa (ex: Fair < Good < Very Good < Premium < Ideal)

**Como carregar:**

```python
import seaborn as sns

df = sns.load_dataset("diamonds")
df.to_csv("data/diamonds.csv", index=False)

# Colunas disponiveis:
# carat (float), cut (category), color (category), clarity (category),
# depth (float), table (float), price (int), x, y, z (float)
print(df.shape)   # (53940, 10)
print(df["cut"].unique())      # ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
print(df["color"].unique())    # ['E', 'I', 'J', 'H', 'F', 'G', 'D']
print(df["clarity"].unique())  # ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']

# Ordem correta para OrdinalEncoder:
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = ["J", "I", "H", "G", "F", "E", "D"]       # D = melhor cor
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
```

**O que praticar:**
- Encoding ordinal com OrdinalEncoder (preserva a ordem das categorias)
- Regressao em dataset maior (~54.000 linhas)
- Avaliacao com RMSE, MAE e R²
- Salvar o pipeline completo de transformacao

**Estrutura proposta — orientada a etapas do pipeline:**

```
diamonds_mlops/
├── pipeline/
│   ├── __init__.py
│   ├── step_01_load.py        # ingestao
│   ├── step_02_transform.py   # encoding ordinal + scaling
│   ├── step_03_train.py       # treino e avaliacao
│   └── step_04_predict.py     # inferencia
├── artifacts/
│   ├── ordinal_encoder.joblib
│   ├── scaler.joblib
│   └── model.joblib
├── data/
│   ├── diamonds.csv
│   └── processed.csv
├── main.py
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --carat 0.89 --cut Premium --color G --clarity SI1 --depth 62.1 --table 58 --x 6.15 --y 6.12 --z 3.81

Preco estimado: $ 4.231
(Intervalo tipico para este perfil: $ 3.800 — $ 4.700)
```

---

## Exercicio D — Eficiencia de Combustivel (MPG)

**Dataset:** `seaborn.load_dataset("mpg")`
**Tarefa:** Regressao — prever o consumo de combustivel em milhas por galao (`mpg`)
**Features sugeridas:** cylinders, displacement, horsepower, weight, acceleration, model_year, origin
**Atenção:** `horsepower` tem alguns valores ausentes; `origin` e categorica (usa, europe, japan)

**Como carregar:**

```python
import seaborn as sns

df = sns.load_dataset("mpg")
df.to_csv("data/mpg.csv", index=False)

# Colunas disponiveis:
# mpg (float), cylinders (int), displacement (float), horsepower (float),
# weight (int), acceleration (float), model_year (int),
# origin (usa/europe/japan), name (str — ignorar)
print(df.shape)   # (398, 9)
print(df.isnull().sum())  # horsepower: 6 nulos
print(df["origin"].value_counts())  # usa: 249, europe: 70, japan: 79

# Remover linhas com mpg ausente antes de treinar
df = df.dropna(subset=["mpg"])
```

**O que praticar:**
- Pipeline com imputacao + encoding + scaling em sequencia
- Usar `sklearn.pipeline.Pipeline` para encadear transformadores (novo conceito)
- Salvar o Pipeline inteiro como um unico `.joblib`
- Inferencia com um unico objeto carregado

**Estrutura proposta — config separado do codigo:**

```
mpg_mlops/
├── config.py          # constantes: features, target, caminhos, hiperparametros
├── src/
│   ├── __init__.py
│   ├── data.py        # download e split
│   ├── model.py       # definicao e treino do pipeline sklearn
│   └── inference.py   # carrega pipeline.joblib e prediz
├── data/
│   ├── mpg.csv
│   └── train_test/
│       ├── train.csv
│       └── test.csv
├── models/
│   └── pipeline.joblib   # Pipeline completo: imputer + encoder + scaler + modelo
├── main.py
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --cylinders 4 --displacement 120 --horsepower 88 --weight 2650 --acceleration 17.5 --model-year 76 --origin japan

Consumo estimado: 30.2 mpg
(Referencia: media do dataset = 23.4 mpg)
```

---

## Exercicio E — Classificador de Cancer de Mama

**Dataset:** `sklearn.datasets.load_breast_cancer()`
**Tarefa:** Classificacao binaria — maligno (1) ou benigno (0)
**Features:** 30 features numericas derivadas de imagens de celulas
**Atenção:** nao ha valores ausentes nem categoricas — o desafio e lidar com muitas features

**Como carregar:**

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

raw = load_breast_cancer()

df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target          # 0 = maligno, 1 = benigno
df.to_csv("data/raw.csv", index=False)

# Informacoes do dataset:
print(df.shape)          # (569, 31)
print(df["target"].value_counts())  # 1: 357 benignos | 0: 212 malignos
print(raw.feature_names)
# ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#  'mean smoothness', ... (30 features no total)]
print(df.isnull().sum().sum())  # 0 — sem valores ausentes
```

**O que praticar:**
- Classificacao com muitas features numericas (alta dimensionalidade relativa)
- Comparar acuracia, precisao, recall e F1 (metricas alem da acuracia)
- Salvar o modelo com suporte a `predict_proba` para retornar probabilidade
- Avaliar curva ROC e salvar o grafico em `reports/`

**Estrutura proposta — orientada a dominios (data / model / evaluation / serving):**

```
cancer_mlops/
├── data/
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
├── model/
│   ├── train.py        # treina e salva
│   ├── evaluate.py     # metricas + curva ROC
│   └── artifacts/
│       ├── scaler.joblib
│       └── classifier.joblib
├── serving/
│   └── predict.py      # inferencia isolada
├── reports/
│   └── roc_curve.png
├── main.py
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --features 17.99 10.38 122.8 1001 0.1184 ... (30 valores)

Diagnostico estimado : Maligno
Probabilidade        : 94.3%

Atencao: este modelo e apenas educacional.
Diagnosticos medicos requerem avaliacao profissional.
```

---

## Exercicio F — Preco de Imoveis (California Housing)

**Dataset:** `sklearn.datasets.fetch_california_housing()`
**Tarefa:** Regressao — prever o valor mediano de imoveis (`MedHouseVal`) por regiao
**Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
**Atenção:** Latitude e Longitude sao features geograficas — sao uteis mas podem precisar de tratamento

**Como carregar:**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

raw = fetch_california_housing()

df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["MedHouseVal"] = raw.target     # valor em centenas de milhares de dolares
df.to_csv("data/housing.csv", index=False)

# Colunas disponiveis:
# MedInc      — renda mediana do bloco (em dezenas de milhares de dolares)
# HouseAge    — idade mediana dos imoveis do bloco
# AveRooms    — media de comodos por residencia
# AveBedrms   — media de quartos por residencia
# Population  — populacao do bloco
# AveOccup    — media de moradores por residencia
# Latitude    — latitude geografica
# Longitude   — longitude geografica
# MedHouseVal — target: valor mediano (em centenas de milhares de USD)
print(df.shape)             # (20640, 9)
print(df.isnull().sum())    # 0 — sem valores ausentes
print(df["MedHouseVal"].describe())
```

**O que praticar:**
- Regressao geografica (features de localizacao)
- Validacao cruzada com K-Fold
- Salvar metricas de treino em `reports/metrics.json` para rastreamento manual
- Estrutura pensando em re-treino futuro (separacao clara de dados e artefatos)

**Estrutura proposta — pensando em re-treino (versionamento de artefatos):**

```
housing_mlops/
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   └── fetch.py          # baixa e salva dataset
│   ├── transform/
│   │   └── features.py       # scaling, salva scaler
│   ├── train/
│   │   └── trainer.py        # treino com CV, salva modelo e metricas
│   └── serve/
│       └── predictor.py      # inferencia isolada
├── data/
│   └── housing.csv
├── artifacts/
│   ├── v1/                   # pasta por versao do modelo
│   │   ├── scaler.joblib
│   │   └── model.joblib
│   └── current -> v1/        # symlink para a versao em uso
├── reports/
│   └── metrics.json          # RMSE, R², data do treino
├── main.py
└── requirements.txt
```

**Saida esperada do predict:**

```
python main.py predict --med-inc 3.5 --house-age 25 --ave-rooms 5.2 --ave-bedrms 1.1 --population 800 --ave-occup 3.1 --latitude 34.05 --longitude -118.25

Valor mediano estimado: $ 198.400
(Unidade original: centenas de milhares de dolares)
Modelo em uso: artifacts/v1/model.joblib
```

---

## Resumo das Estruturas

| Exercicio | Dataset | Tarefa | Estrutura |
|-----------|---------|--------|-----------|
| A — Titanic | seaborn | Classificacao | Flat — scripts na raiz |
| B — Tips | seaborn | Regressao | src/ plano sem subdiretorios |
| C — Diamonds | seaborn | Regressao | Orientada a etapas numeradas |
| D — MPG | seaborn | Regressao | Config separado + Pipeline sklearn |
| E — Cancer | sklearn | Classificacao | Orientada a dominios (data/model/serving) |
| F — Housing | sklearn | Regressao | Versionamento de artefatos por pasta |

---

## Dica Geral

Independente da estrutura escolhida, os tres artefatos fundamentais sempre existem:

```
1. Transformadores ajustados no treino  (imputer, scaler, encoders)
2. Modelo treinado                      (model.joblib)
3. Lista de features na ordem correta   (constante no codigo ou em config)
```

Sem esses tres, a inferencia nao consegue reproduzir o que o treino fez.
