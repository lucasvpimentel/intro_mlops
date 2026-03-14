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

**Sugestoes de prompts:**

> **download.py**
> "Crie um script Python chamado download.py que carrega o dataset Titanic usando seaborn.load_dataset('titanic'), seleciona apenas as colunas pclass, sex, age, sibsp, parch, fare, embarked e survived, e salva em data/raw.csv. Crie a pasta data/ se nao existir."

> **prepare.py**
> "Crie um script Python chamado prepare.py que le data/raw.csv e faz o seguinte: (1) imputa valores ausentes de 'age' com a mediana usando SimpleImputer, (2) imputa 'embarked' com o valor mais frequente, (3) aplica LabelEncoder em 'sex' e 'embarked', (4) salva o dataset processado em data/processed.csv, (5) salva os transformadores (imputer_num.joblib, imputer_cat.joblib, le_sex.joblib, le_embarked.joblib) em data/models/ usando joblib. Todos os transformadores devem ser ajustados apenas nos dados de treino."

> **train.py**
> "Crie um script Python chamado train.py que le data/processed.csv, separa em treino (80%) e teste (20%) com random_state=42 e stratify=survived, treina um RandomForestClassifier com n_estimators=100 e random_state=42, avalia com accuracy, precision, recall e F1, e salva o modelo em data/models/model.joblib."

> **predict.py**
> "Crie uma funcao predict(pclass, sex, age, sibsp, parch, fare, embarked) que: carrega os transformadores salvos (imputer, label encoders) e o modelo de data/models/, aplica as mesmas transformacoes do treino nos valores de entrada, retorna um dicionario com 'sobreviveu' (bool) e 'probabilidade' (float de 0 a 100). Adicione if __name__ == '__main__' com um exemplo de uso."

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

**Sugestoes de prompts:**

> **src/loader.py**
> "Crie um modulo Python src/loader.py com uma funcao load() que carrega o dataset Tips usando seaborn.load_dataset('tips') e salva em data/tips.csv. A funcao deve retornar o DataFrame."

> **src/features.py**
> "Crie um modulo src/features.py com uma funcao build(modo_treino=True) que: (1) le data/tips.csv, (2) aplica LabelEncoder nas colunas sex, smoker, day e time, (3) aplica StandardScaler nas colunas total_bill e size, (4) no modo_treino=True, ajusta e salva os transformadores em data/artifacts/ com joblib, (5) no modo_treino=False, carrega os transformadores salvos. Retorna X (features) e y (tip)."

> **src/trainer.py**
> "Crie um modulo src/trainer.py com uma funcao train(model_type='ridge') que treina Ridge ou RandomForestRegressor no dataset processado pelo src/features.py, avalia com RMSE e R² usando cross_val_score com 5 folds, e salva o modelo em data/artifacts/model.joblib. Exiba os resultados no terminal."

> **src/predictor.py**
> "Crie um modulo src/predictor.py com uma funcao predict(total_bill, sex, smoker, day, time, size) que carrega os transformadores de data/artifacts/ e o model.joblib, aplica as mesmas transformacoes nos valores de entrada, e retorna o valor estimado da gorjeta como float. Adicione if __name__ == '__main__' com um exemplo."

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

**Sugestoes de prompts:**

> **pipeline/step_01_load.py**
> "Crie um script pipeline/step_01_load.py com uma funcao load() que carrega o dataset Diamonds usando seaborn.load_dataset('diamonds'), remove linhas duplicadas, salva em data/diamonds.csv e retorna o DataFrame com shape e dtypes impressos no terminal."

> **pipeline/step_02_transform.py**
> "Crie um script pipeline/step_02_transform.py com uma funcao transform(modo_treino=True) que aplica OrdinalEncoder nas colunas cut, color e clarity com as seguintes ordens: cut=['Fair','Good','Very Good','Premium','Ideal'], color=['J','I','H','G','F','E','D'], clarity=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']. Em seguida aplica StandardScaler em todas as features numericas. No modo_treino=True salva os transformadores em artifacts/ com joblib. Retorna X e y (price)."

> **pipeline/step_03_train.py**
> "Crie um script pipeline/step_03_train.py com uma funcao train() que carrega os dados transformados pelo step_02, treina um RandomForestRegressor com n_estimators=200 e random_state=42, avalia com RMSE, MAE e R² no conjunto de teste (20%), e salva o modelo em artifacts/model.joblib. Exiba as metricas formatadas no terminal."

> **pipeline/step_04_predict.py**
> "Crie um script pipeline/step_04_predict.py com uma funcao predict(carat, cut, color, clarity, depth, table, x, y, z) que carrega o OrdinalEncoder, o StandardScaler e o modelo de artifacts/, aplica as transformacoes na entrada e retorna o preco estimado como float. Adicione if __name__ == '__main__' com um diamante de exemplo."

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

**Sugestoes de prompts:**

> **config.py**
> "Crie um arquivo config.py com as seguintes constantes: FEATURES_NUM (lista com cylinders, displacement, horsepower, weight, acceleration, model_year), FEATURES_CAT (lista com origin), TARGET ('mpg'), RANDOM_STATE (42), TEST_SIZE (0.2), N_ESTIMATORS (100), e os caminhos DATA_PATH, TRAIN_PATH, TEST_PATH e MODEL_PATH usando pathlib.Path."

> **src/data.py**
> "Crie um modulo src/data.py com uma funcao load_and_split() que: carrega o dataset MPG via seaborn.load_dataset('mpg'), remove a coluna 'name', remove linhas onde 'mpg' e nulo, salva o dataset completo em data/mpg.csv, divide em treino e teste usando os valores de config.py e salva os splits em data/train_test/."

> **src/model.py**
> "Crie um modulo src/model.py com uma funcao build_pipeline() que retorna um sklearn.pipeline.Pipeline com tres etapas: (1) ColumnTransformer que aplica SimpleImputer(strategy='median') nas features numericas e OneHotEncoder(handle_unknown='ignore') em 'origin', (2) StandardScaler, (3) RandomForestRegressor com parametros de config.py. Crie tambem uma funcao train() que ajusta o pipeline nos dados de treino, avalia com RMSE e R², e salva o pipeline inteiro em models/pipeline.joblib."

> **src/inference.py**
> "Crie um modulo src/inference.py com uma funcao predict(cylinders, displacement, horsepower, weight, acceleration, model_year, origin) que carrega models/pipeline.joblib, monta um DataFrame de uma linha com os valores recebidos e retorna o consumo estimado em mpg. O pipeline ja inclui imputacao, encoding e scaling — nao e necessario pre-processar manualmente. Adicione if __name__ == '__main__' com um carro de exemplo."

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

**Sugestoes de prompts:**

> **data/fetch.py (ou equivalente)**
> "Crie um script que carrega o dataset Breast Cancer usando sklearn.datasets.load_breast_cancer(), cria um DataFrame com as 30 features e a coluna 'target' (0=maligno, 1=benigno), salva em data/raw.csv e divide em data/train.csv e data/test.csv com stratify=target e random_state=42."

> **model/train.py**
> "Crie um script model/train.py que le data/train.csv, aplica StandardScaler nas 30 features numericas, treina um RandomForestClassifier com n_estimators=200 e random_state=42, e salva o scaler em model/artifacts/scaler.joblib e o modelo em model/artifacts/classifier.joblib."

> **model/evaluate.py**
> "Crie um script model/evaluate.py que le data/test.csv, carrega scaler e classifier de model/artifacts/, avalia com accuracy, precision, recall e F1-score usando sklearn.metrics, imprime o classification_report completo, gera a curva ROC com matplotlib e salva em reports/roc_curve.png."

> **serving/predict.py**
> "Crie um modulo serving/predict.py com uma funcao predict(features: list) que recebe uma lista com exatamente 30 valores numericos, carrega model/artifacts/scaler.joblib e model/artifacts/classifier.joblib, aplica o scaler e retorna um dicionario com 'diagnostico' ('Maligno' ou 'Benigno') e 'probabilidade' (float de 0 a 100). Adicione if __name__ == '__main__' com a primeira linha do dataset como exemplo."

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

**Sugestoes de prompts:**

> **src/ingest/fetch.py**
> "Crie um script src/ingest/fetch.py com uma funcao fetch() que carrega o California Housing dataset usando sklearn.datasets.fetch_california_housing(), cria um DataFrame com as 8 features e a coluna 'MedHouseVal' como target, e salva em data/housing.csv. Imprima shape, dtypes e describe() no terminal."

> **src/transform/features.py**
> "Crie um modulo src/transform/features.py com uma funcao transform(modo_treino=True, version='v1') que: le data/housing.csv, separa em treino e teste (80/20, random_state=42), aplica StandardScaler apenas nas features numericas (todas as 8 colunas sao numericas), e salva o scaler em artifacts/{version}/scaler.joblib. Retorna X_train, X_test, y_train, y_test."

> **src/train/trainer.py**
> "Crie um modulo src/train/trainer.py com uma funcao train(version='v1') que chama transform(), treina um RandomForestRegressor com n_estimators=200 e random_state=42, avalia com K-Fold (k=5) calculando RMSE e R², salva o modelo em artifacts/{version}/model.joblib, e salva as metricas (RMSE, R², data, versao) em reports/metrics.json."

> **src/serve/predictor.py**
> "Crie um modulo src/serve/predictor.py com uma funcao predict(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, version='v1') que carrega artifacts/{version}/scaler.joblib e artifacts/{version}/model.joblib, aplica o scaler e retorna o valor mediano estimado em dolares (multiplique por 100.000 para converter da unidade original). Adicione if __name__ == '__main__' com coordenadas de Los Angeles como exemplo."

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
