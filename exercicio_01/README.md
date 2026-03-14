# Exercicio 01 — Classificador de Especies Iris

Classificador de flores Iris usando Regressao Logistica. Treinado com
`scikit-learn`, persistido com `joblib` e exposto via `main.py`.

## Estrutura do projeto

```
exercicio_01/
├── data/
│   ├── raw.csv               # Dataset gerado por download_data.py
│   └── iris_model.joblib     # Modelo gerado por train.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_data.py  # Baixa e salva o dataset
│   └── models/
│       ├── __init__.py
│       ├── train.py          # Treina e serializa o modelo
│       └── predict.py        # Logica de inferencia
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
| `scikit-learn` | Dataset Iris, Regressao Logistica, metricas |
| `pandas` | Manipulacao do CSV |
| `joblib` | Serializar/carregar o modelo |
| `numpy` | Operacoes numericas (dependencia interna) |

### Passo 3 — Executar o pipeline completo

```bash
python main.py pipeline
```

Executa download + treino em sequencia. Saida esperada:

```
==> [1/2] Baixando dataset...
Dataset salvo em: .../data/raw.csv
Shape: (150, 6)

==> [2/2] Treinando modelo...
=== Relatorio de Classificacao ===
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10
    accuracy                           0.97        30

Modelo salvo em: .../data/iris_model.joblib
```

Ou execute cada etapa individualmente:

```bash
python main.py download   # Baixa o dataset
python main.py train      # Treina o modelo
```

### Passo 4 — Prever uma nova flor

```bash
python main.py predict <sepal_length> <sepal_width> <petal_length> <petal_width>
```

**Exemplos:**

| Especie esperada | Comando |
|---|---|
| setosa     | `python main.py predict 5.1 3.5 1.4 0.2` |
| versicolor | `python main.py predict 6.0 2.7 5.1 1.6` |
| virginica  | `python main.py predict 6.7 3.3 5.7 2.5` |

Saida:
```
Especie prevista : setosa
Confianca        : 97.8%
```

---

## Artefatos Salvos e Uso Independente

Apos rodar `train` (ou `pipeline`) uma vez, o modelo fica persistido em disco:

```
data/
└── iris_model.joblib   # unico artefato necessario para prever
```

**Voce nao precisa retreinar para fazer predicoes.** Com o `venv` ja criado e
o modelo ja treinado, basta ativar e rodar o `predict`:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Prever — nao toca em dados de treino, nao retreina nada
python main.py predict 5.1 3.5 1.4 0.2    # setosa
python main.py predict 6.0 2.7 5.1 1.6    # versicolor
python main.py predict 6.7 3.3 5.7 2.5    # virginica
```

O `predict` executa apenas `src/models/predict.py`, que faz:
1. `joblib.load("data/iris_model.joblib")` — carrega o modelo salvo
2. `model.predict(...)` — faz a predicao em memoria
3. Retorna especie + confianca

Se voce deletar `data/raw.csv`, o `predict` ainda funciona — ele so depende
do `.joblib`. Os dados de treino nao sao necessarios para inferencia.

---

## Como funciona internamente

```
main.py
  |-- download  --> src/data/download_data.py  --> data/raw.csv
  |-- train     --> src/models/train.py        --> data/iris_model.joblib
  |-- predict   --> src/models/predict.py      --> carrega modelo, retorna especie
  |-- pipeline  --> download + train
```

## Decisoes de projeto

- **Regressao Logistica:** modelo linear simples, interpretavel e suficiente
  para o Iris Dataset (linearmente separavel em 2 das 3 classes).
- **`joblib`:** mais eficiente que `pickle` para objetos numpy/sklearn.
- **DataFrame na inferencia:** evita warning de feature names do sklearn ao
  passar os dados como `numpy array`.
- **Estratificacao no split:** garante proporcao igual das 3 classes no
  conjunto de teste.
- **`src/` como modulo:** todos os subdiretorios possuem `__init__.py`,
  permitindo imports limpos via `from src.models.predict import predict`.
