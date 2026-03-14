# Exercicio Final — Penguins MLOps

Sistema de dupla predicao para pinguins da Estacao Palmer:

- **Tarefa A (Classificacao)**: identificar a especie (Adelie, Chinstrap ou Gentoo)
- **Tarefa B (Regressao)**: estimar a massa corporal (peso em gramas)

Algoritmo: **Random Forest** (proibido o uso de Redes Neurais).

---

## Estrutura do Projeto

```
exercicio_final/
├── data/
│   ├── raw/               # penguins.csv (dado original, nao modificar)
│   ├── processed/         # train.csv e test.csv (gerados pelo split)
│   └── samples/           # new_penguins.json (amostras para inferencia)
├── docs/
│   └── dicionario_atributos.md   # Descricao de todas as colunas
├── models/                # Artefatos salvos: classifier.joblib, regressor.joblib,
│                          # scaler.joblib, le_sex.joblib, le_island.joblib, le_species.joblib
├── notebooks/             # Analise exploratoria (EDA)
├── reports/               # Graficos: confusion_matrix.png, weight_error_by_species.png
├── scripts/
│   └── run_pipeline.bat   # Script Windows para executar o pipeline completo
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Download e divisao treino/teste
│   ├── preprocessor.py    # Imputacao, encoding, normalizacao
│   ├── trainer.py         # Treinamento dos dois modelos + CV
│   ├── evaluator.py       # Metricas + graficos no conjunto de teste
│   └── inference.py       # Predicao individual e em lote
├── main.py                # Orquestrador CLI (ponto de entrada unico)
└── requirements.txt
```

---

## Configuracao do Ambiente

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar (Windows)
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Plano de Implementacao Passo a Passo

### Passo 1 — Baixar o dataset

```bash
python main.py download
```

Salva o Palmer Penguins Dataset em `data/raw/penguins.csv` (344 registros, 7 colunas).

### Passo 2 — Dividir em treino e teste

```bash
python main.py split
```

Divide em 80% treino / 20% teste com estratificacao por especie.
Salva `data/processed/train.csv` e `data/processed/test.csv`.

### Passo 3 — Treinar os modelos

```bash
python main.py train
```

Executa as seguintes etapas:

1. Carrega `data/processed/train.csv`
2. Aplica preprocessamento completo (modo treino):
   - Imputacao: media para numericas, moda para categoricas
   - Encoding: `sex` e `island` → LabelEncoder
   - Normalizacao: StandardScaler nas 5 features
3. Treina `RandomForestClassifier(n_estimators=200)` para especie
4. Avalia com cross-validation 5-fold (metrica: accuracy)
5. Treina `RandomForestRegressor(n_estimators=200)` para peso
6. Avalia com cross-validation 5-fold (metrica: RMSE)
7. Salva todos os artefatos em `models/`

### Passo 4 — Avaliar no conjunto de teste

```bash
python main.py evaluate
```

Gera:
- Accuracy, Precision, Recall, F1 por especie
- Matriz de confusao → `reports/confusion_matrix.png`
- RMSE, MAE, R² da regressao
- Grafico de erro de peso por especie → `reports/weight_error_by_species.png`

### Passo 5 — Realizar predicoes

**Predicao individual** (um pinguim medido em campo):

```bash
python main.py predict \
  --bill-length 39.1 \
  --bill-depth 18.7 \
  --flipper-length 181.0 \
  --sex male \
  --island Torgersen
```

**Predicao em lote** (a partir de `data/samples/new_penguins.json`):

```bash
python main.py predict-batch
```

---

## Pipeline Completo (todos os passos de uma vez)

```bash
python main.py pipeline
```

Ou usando o script Windows:

```
scripts\run_pipeline.bat
```

---

## Entendendo o Preprocessamento

O `preprocessor.py` tem dois modos:

| Modo          | Quando usar        | O que faz                                           |
|---------------|--------------------|-----------------------------------------------------|
| `modo_treino=True`  | Durante o treino   | Fit + transform + salva artefatos em `models/`      |
| `modo_treino=False` | Durante inferencia | Carrega artefatos + aplica transform (sem fit)      |

**Por que isso e importante?**
Se fizermos fit na inferencia, a "media" que o scaler usa seria a media dos dados novos,
nao a media dos dados de treino. Isso causaria predicoes erradas e inconsistentes.

### Artefatos salvos em `models/`

| Arquivo               | Proposito                                          |
|-----------------------|----------------------------------------------------|
| `classifier.joblib`   | RandomForestClassifier treinado                    |
| `regressor.joblib`    | RandomForestRegressor treinado                     |
| `scaler.joblib`       | StandardScaler com media/std do treino             |
| `le_sex.joblib`       | LabelEncoder para `sex` (male/female → 0/1)       |
| `le_island.joblib`    | LabelEncoder para `island` (Biscoe/Dream/... → 0/1/2) |
| `le_species.joblib`   | LabelEncoder para `species` (para decodificar predicao) |
| `imputer_num.joblib`  | SimpleImputer para features numericas              |
| `imputer_cat.joblib`  | SimpleImputer para features categoricas            |

---

## Artefatos Salvos e Uso Independente

Apos rodar `train` (ou `pipeline`) uma vez, todos os artefatos ficam persistidos:

```
models/
├── classifier.joblib    # RandomForestClassifier (especie)
├── regressor.joblib     # RandomForestRegressor (peso)
├── scaler.joblib        # StandardScaler (media/std aprendidos no treino)
├── le_sex.joblib        # LabelEncoder: Male/Female -> 0/1
├── le_island.joblib     # LabelEncoder: Biscoe/Dream/Torgersen -> 0/1/2
├── le_species.joblib    # LabelEncoder: para decodificar a predicao
├── imputer_num.joblib   # SimpleImputer para features numericas
└── imputer_cat.joblib   # SimpleImputer para features categoricas
```

**Voce nao precisa retreinar para fazer predicoes.** Com o `venv` ja criado e
os modelos ja treinados, basta ativar e rodar o `predict` ou `predict-batch`:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2a. Predicao individual — mede um pinguim de campo
python main.py predict \
  --bill-length 39.1 \
  --bill-depth 18.7 \
  --flipper-length 181.0 \
  --sex male \
  --island Torgersen
```

```
========================================
  PREDICAO
========================================
  Especie        : Adelie
  Confianca      : 100.0%
  Peso estimado  : 3727.0 g
========================================
```

```bash
# 2b. Predicao em lote — classifica varios pinguins de uma vez
#     Edite data/samples/new_penguins.json com as medicoes do dia
python main.py predict-batch
```

```
Processando 5 pinguins de data/samples/new_penguins.json
  [ 1] Adelie       100.0%     3727g
  [ 2] Chinstrap     99.5%     3478g
  [ 3] Gentoo       100.0%     5186g
  ...
```

O `predict` e o `predict-batch` executam apenas `src/inference.py`, que faz:
1. `joblib.load("models/classifier.joblib")` — carrega o classificador
2. `joblib.load("models/regressor.joblib")` — carrega o regressor
3. Aplica `preprocessor.py` com `modo_treino=False` — carrega e aplica
   os scalers/encoders salvos, sem nenhum `fit` novo
4. Retorna especie + confianca + peso estimado

Se voce deletar `data/` inteiro (dados brutos e processados), o `predict`
ainda funciona — ele so depende dos `.joblib` em `models/`.

---

## Tres Principios Aplicados

### Limpeza
`data_loader.py` entrega os dados prontos (sem nulos nos alvos) antes do preprocessamento.
`preprocessor.py` centraliza todas as transformacoes — `trainer.py` e `inference.py` nao processam dados diretamente.

### Reprodutibilidade
`train_test_split` usa `stratify=df["species"]` e `random_state=42`.
`RandomForest` usa `random_state=42`. O mesmo comando sempre produz os mesmos artefatos.

### Isolamento
`inference.py` so carrega arquivos `.joblib` — nunca acessa dados de treino.
O preprocessamento na inferencia usa `modo_treino=False`, garantindo que os
transformadores aprendidos no treino sejam reutilizados sem modificacao.

---

## Metricas Esperadas

| Tarefa          | Metrica     | Valor tipico     |
|-----------------|-------------|-----------------|
| Classificacao   | Accuracy    | > 97%           |
| Classificacao   | F1 (macro)  | > 97%           |
| Regressao       | RMSE        | < 350 g         |
| Regressao       | R²          | > 0.85          |
