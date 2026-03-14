# Exercicio 02 — Estimador de Progressao de Doenca (Diabetes Dataset)

Regressao sobre o Diabetes Dataset do sklearn. O ponto central e a persistencia
do `StandardScaler` para garantir que a inferencia usa a mesma normalizacao
do treino.

## Estrutura do projeto

```
exercicio_02/
├── data/
│   ├── raw.csv               # Dataset bruto
│   ├── processed.csv         # Dataset normalizado
│   ├── scaler.joblib         # StandardScaler ajustado no treino
│   └── model.joblib          # Modelo treinado (Ridge ou Random Forest)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_data.py    # Baixa e salva o dataset
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Normaliza features e salva o scaler
│   └── models/
│       ├── __init__.py
│       ├── train.py            # Treina e serializa o modelo
│       └── predict.py          # Logica de inferencia
├── main.py                   # Ponto de entrada unico
└── requirements.txt
```

---

## Plano de construcao manual (passo a passo)

### Passo 1 — Criar o ambiente virtual

```bash
python -m venv venv
```

Ativar o ambiente:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### Passo 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

| Pacote | Finalidade |
|---|---|
| `scikit-learn` | Dataset, modelos, StandardScaler, metricas |
| `pandas` | Manipulacao de CSVs |
| `joblib` | Serializar scaler e modelo |
| `numpy` | Operacoes numericas |

### Passo 3 — Executar o pipeline completo

```bash
python main.py pipeline
```

Executa download + normalizacao + treino em sequencia. Saida esperada:

```
==> [1/3] Baixando dataset...
Dataset salvo em: .../data/raw.csv
Shape: (442, 11)

==> [2/3] Normalizando features e salvando scaler...
Scaler salvo em: .../data/scaler.joblib
Dataset normalizado salvo em: .../data/processed.csv

==> [3/3] Treinando modelo (ridge)...
CV R2 (5-fold): 0.4512 +/- 0.1388

=== RIDGE - Conjunto de Teste ===
  RMSE : 53.78
  MAE  : 42.81
  R2   : 0.4541
```

Ou execute cada etapa individualmente:

```bash
python main.py download              # Baixa o dataset
python main.py features              # Normaliza e salva o scaler
python main.py train                 # Ridge (padrao)
python main.py train --model rf      # Random Forest
```

### Passo 4 — Normalizar features e salvar o scaler

Este passo e o centro do exercicio. `main.py features` executa
`src/features/build_features.py`, que:

1. Le `data/raw.csv`
2. Ajusta um `StandardScaler` nas 10 features
3. Salva o scaler em `data/scaler.joblib`
4. Salva o dataset normalizado em `data/processed.csv`

> **Por que salvar o scaler?**
> Na inferencia, os dados de entrada estao em escala bruta. Sem aplicar
> o mesmo scaler do treino, os valores entrariam no modelo em distribuicao
> diferente, gerando predicoes incorretas.

### Passo 5 — Prever para um novo paciente

```bash
python main.py predict <age> <sex> <bmi> <bp> <s1> <s2> <s3> <s4> <s5> <s6>
```

A inferencia executa em 3 etapas (via `src/models/predict.py`):
1. Carrega `scaler.joblib` e normaliza os valores de entrada
2. Carrega `model.joblib`
3. Retorna o indice estimado de progressao

**Exemplo:**
```bash
python main.py predict 0.038 0.050 0.061 0.021 -0.044 -0.034 -0.043 -0.002 0.019 -0.017
```
```
Progressao estimada da diabetes (1 ano): 209.1
(Escala: ~25 = baixa progressao | ~346 = alta progressao)
```

> **Por que os valores parecem pequenos?**
> O `sklearn.datasets.load_diabetes()` ja entrega as features pre-padronizadas
> (media zero, escala ~[-0.2, +0.2]). Os valores acima sao da escala original
> do dataset — nao representam idade em anos ou IMC em kg/m². O scaler interno
> aplica uma normalizacao adicional sobre esses valores antes de passar ao modelo.

---

## Artefatos Salvos e Uso Independente

Apos rodar `pipeline` uma vez, dois artefatos ficam persistidos em disco:

```
data/
├── scaler.joblib   # StandardScaler ajustado nos dados de treino
└── model.joblib    # modelo treinado (Ridge ou Random Forest)
```

**Voce nao precisa retreinar para fazer predicoes.** Com o `venv` ja criado e
os artefatos ja gerados, basta ativar e rodar o `predict`:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Prever — so carrega scaler.joblib e model.joblib, nao retreina nada
# Os valores abaixo estao na escala pre-padronizada do sklearn diabetes dataset
python main.py predict 0.038 0.050 0.061 0.021 -0.044 -0.034 -0.043 -0.002 0.019 -0.017
```

O `predict` executa apenas `src/models/predict.py`, que faz:
1. `joblib.load("data/scaler.joblib")` — carrega o scaler salvo
2. `scaler.transform(entrada)` — normaliza os dados de entrada com a escala do treino
3. `joblib.load("data/model.joblib")` — carrega o modelo salvo
4. `model.predict(...)` — retorna o indice de progressao estimado

> **Por que o scaler e obrigatorio na inferencia?**
> Os dados brutos de entrada estao em escala original (ex: bmi = 0.06).
> O modelo foi treinado com dados normalizados. Sem aplicar o mesmo scaler,
> a predicao seria errada — o modelo veria valores em uma escala diferente
> da que aprendeu.

Se voce deletar `data/raw.csv` e `data/processed.csv`, o `predict` ainda
funciona — ele so depende dos dois `.joblib`.

---

## Como funciona internamente

```
main.py
  |-- download  --> src/data/download_data.py    --> data/raw.csv
  |-- features  --> src/features/build_features  --> data/scaler.joblib
  |                                                  data/processed.csv
  |-- train     --> src/models/train.py          --> data/model.joblib
  |-- predict   --> src/models/predict.py        --> carrega scaler + modelo,
  |                                                  normaliza entrada,
  |                                                  retorna predicao
  |-- pipeline  --> download + features + train
```

## Decisoes de projeto

- **Ridge Regression como padrao:** regularizacao L2 adequada para datasets
  pequenos com features correlacionadas (como as sanguineas s1-s6).
- **Random Forest como alternativa:** captura relacoes nao-lineares, util
  para comparar com o modelo linear.
- **Scaler separado do modelo:** permite trocar o modelo sem re-ajustar o
  scaler, e vice-versa.
- **`processed.csv` gerado:** permite auditar os valores normalizados sem
  re-executar o scaler.
- **`src/` como modulo:** todos os subdiretorios possuem `__init__.py`,
  permitindo imports limpos via `from src.features.build_features import build`.
