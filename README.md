# Intro MLOps — Exercicios Praticos

Repositorio com 7 exercicios progressivos cobrindo o ciclo completo de um
projeto de Machine Learning: desde a preparacao dos dados ate o monitoramento
de drift em producao.

---

## Estrutura dos Exercicios

| Pasta | Tema | Algoritmo | Conceito central |
|---|---|---|---|
| `exercicio_01` | Classificador Iris | Logistic Regression | Persistencia de modelo com joblib |
| `exercicio_02` | Regressao Diabetes | Ridge / Random Forest | Persistencia de scaler + modelo |
| `exercicio_03` | Classificador Vinhos | Random Forest | Batch inference + avaliacao separada |
| `exercicio_04` | Drift Monitor Iris | KS Test | Deteccao de data drift |
| `exercicio_05` | Drift Monitor Diabetes | KS Test + PSI | PSI como metrica de deslocamento |
| `exercicio_06` | Drift Monitor Vinhos | KS + PSI + JSD | Sistema de votos multi-metodo |
| `exercicio_final` | Pinguins MLOps | Random Forest (2 tarefas) | Pipeline completo multi-output |

---

## Teoria Aplicada nos Exercicios

### 1. O Ciclo de Vida de um Modelo ML

Todo projeto de ML em producao segue um ciclo com quatro fases principais:

```
[Dados] --> [Treino] --> [Avaliacao] --> [Producao] --> [Monitoramento]
                ^                                              |
                |______________ Retreino _____________________|
```

Os exercicios 01-03 cobrem as tres primeiras fases.
Os exercicios 04-06 cobrem o monitoramento.
O exercicio_final integra todas as fases.

---

### 2. Persistencia de Modelos com joblib

Apos o treino, um modelo sklearn e um objeto Python em memoria. Para usa-lo
sem retreinar, ele precisa ser serializado (salvo em disco).

```python
import joblib

# Treino: salva o modelo
joblib.dump(model, "modelo.joblib")

# Inferencia: carrega o modelo
model = joblib.load("modelo.joblib")
predicao = model.predict(X_novo)
```

**Por que joblib e nao pickle?**
`joblib` e mais eficiente para objetos que contem arrays numpy (como os
RandomForest, que internamente tem centenas de arvores como arrays). Para
objetos grandes, `joblib` e tipicamente 2-10x mais rapido e gera arquivos menores.

**O que DEVE ser salvo alem do modelo:**

| Artefato | Por que salvar |
|---|---|
| `model.joblib` | O proprio modelo treinado |
| `scaler.joblib` | Os parametros de normalizacao (media e std do treino) |
| `le_*.joblib` | Os mapeamentos de categorias (ex: "Male" -> 1) |
| `imputer.joblib` | A estrategia de imputacao ajustada nos dados de treino |

Se voce nao salvar o scaler, a inferencia aplicaria uma normalizacao diferente
e o modelo receberia dados em uma distribuicao que nunca viu durante o treino.

---

### 3. StandardScaler e o Problema da Escala

Muitos algoritmos (Ridge, SVM, KNN) sao sensiveis a escala das features.
Uma feature com valores em [0, 1000] domina o aprendizado sobre uma com [0, 1].

O `StandardScaler` normaliza cada feature para media=0 e desvio padrao=1:

```
X_normalizado = (X - media_treino) / std_treino
```

**Regra fundamental:** o `fit()` so acontece nos dados de treino. Na inferencia,
apenas o `transform()` e aplicado com os parametros aprendidos no treino.

```python
# CERTO: aprende no treino, aplica no teste/inferencia
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # so transform!

# ERRADO: re-aprende no teste (data leakage)
X_test_scaled = scaler.fit_transform(X_test)  # NUNCA faca isso
```

---

### 4. Validacao Cruzada (Cross-Validation)

Um unico split treino/teste pode ser enganoso: o modelo pode ter "sorte" com
aquele conjunto de teste especifico. A validacao cruzada k-fold resolve isso:

```
Fold 1: [TESTE] [treino] [treino] [treino] [treino]
Fold 2: [treino] [TESTE] [treino] [treino] [treino]
Fold 3: [treino] [treino] [TESTE] [treino] [treino]
...
```

O modelo e treinado k vezes, cada vez com um subconjunto diferente como teste.
A metrica final e a media dos k resultados — muito mais confiavel que uma unica avaliacao.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

O `+/- std` e tao importante quanto a media: um modelo com accuracy 0.95 +/- 0.15
e menos confiavel que um com 0.92 +/- 0.02.

---

### 5. Tres Principios de Codigo MLOps

Todos os exercicios seguem tres principios de organizacao:

#### Limpeza
O modulo de dados (`src/data/`) entrega os dados prontos para o modelo.
O modulo de modelos (`src/models/`) nao faz limpeza — confia no que recebe.

```
download_data.py  -->  build_features.py  -->  train.py
     (raw)               (processed)           (model)
```

#### Reprodutibilidade
Deletar `data/models/` e `data/processed/` e rodar `python main.py pipeline`
deve sempre produzir exatamente o mesmo modelo.

Isso exige:
- `random_state=42` em todos os splits e modelos
- `stratify=y` no `train_test_split` para datasets com classes desbalanceadas
- Dependencias fixadas no `requirements.txt` com versoes exatas

#### Isolamento
`predict.py` (ou `inference.py`) so le arquivos `.joblib` — nunca acessa
dados de treino, nunca re-treina, nunca re-normaliza a partir dos dados.

```python
# CERTO: isolamento total
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
pred   = model.predict(scaler.transform(X_novo))

# ERRADO: vazamento para dentro da inferencia
df = pd.read_csv("data/raw.csv")   # nunca leia dados de treino na inferencia
scaler.fit(df[features])           # nunca refaca o fit na inferencia
```

---

### 6. Data Drift — O Problema da Producao

Um modelo treinado hoje pode se degradar ao longo do tempo sem nenhum erro
explicito. A causa mais comum e **data drift**: os dados de entrada mudaram
em relacao a distribuicao usada no treino.

#### Tipos de Drift

| Tipo | Definicao | Exemplo |
|---|---|---|
| **Covariate shift** | Distribuicao de X mudou, P(Y\|X) estavel | Perfil de clientes mudou |
| **Label drift** | Distribuicao de Y mudou | Sazonalidade nos alvos |
| **Concept drift** | A relacao P(Y\|X) mudou | Comportamento dos usuarios mudou |

Os exercicios 04-06 focam em covariate shift (mudanca nas features de entrada).

#### Por que o modelo se degrada?

O modelo aprendeu `P(Y | X_treino)`. Se `X_producao` tem distribuicao diferente
de `X_treino`, o modelo esta sendo aplicado fora do dominio que conhece.

```
X_treino:    media=170cm, std=10cm  (populacao de referencia)
X_producao:  media=185cm, std=10cm  (nova populacao, 15cm acima)

O modelo nunca viu valores nessa faixa com aquele significado.
```

---

### 7. Teste de Kolmogorov-Smirnov (KS)

O teste KS de duas amostras verifica se dois conjuntos de dados vierem da mesma
distribuicao comparando suas funcoes de distribuicao acumulada (CDF).

```
CDF_referencia(x) = fracao dos valores de referencia <= x
CDF_nova(x)       = fracao dos valores novos <= x

Estatistica KS = max |CDF_referencia(x) - CDF_nova(x)|
```

- **KS = 0**: distribuicoes identicas
- **KS = 1**: distribuicoes completamente separadas
- **p-value < 0.05**: a diferenca e estatisticamente significativa (drift detectado)

```python
from scipy import stats
ks_stat, p_value = stats.ks_2samp(referencia, novos)
drift = p_value < 0.05
```

**Vantagens:** nao-parametrico (nao assume normalidade), sensivel a qualquer
diferenca de forma, localizacao ou escala.

**Limitacao:** resposta binaria (drift/nao-drift) sem indicar a magnitude.

---

### 8. PSI — Population Stability Index

O PSI quantifica o quanto a distribuicao atual desviou da referencia usando
histogramas com bins fixos. E a metrica padrao em modelos de credito bancario.

```
PSI = sum( (atual% - esperado%) * ln(atual% / esperado%) )
```

Onde `atual%` e `esperado%` sao as proporcoes de amostras em cada bin.

| PSI | Status | Interpretacao |
|---|---|---|
| < 0.10 | OK | Distribuicao estavel |
| 0.10 – 0.20 | WARNING | Mudanca moderada — monitorar |
| >= 0.20 | ALERT | Mudanca severa — investigar e possivelmente retreinar |

**Vantagem sobre o KS:** numerico e interpretavel. PSI = 0.35 diz que houve
"muito mais drift" que PSI = 0.12, algo que o KS nao consegue expressar.

**Como e calculado:**
1. Define bins baseados nos percentis da referencia (garantindo bins populados)
2. Calcula proporcoes da referencia por bin (esperado)
3. Calcula proporcoes dos dados novos nos mesmos bins (atual)
4. Aplica a formula com `clip` nos valores fora do range (padrao de producao)

---

### 9. Jensen-Shannon Divergence (JSD)

A JSD e uma medida simetrica de distancia entre duas distribuicoes de
probabilidade, baseada na divergencia de Kullback-Leibler:

```
JSD(P, Q) = (KL(P||M) + KL(Q||M)) / 2    onde M = (P + Q) / 2
KL(P||Q)  = sum( P * log(P/Q) )           -- assimetrica, pode ser infinita
```

**Por que JSD e melhor que KL em monitoramento:**

| Propriedade | KL Divergence | JS Divergence |
|---|---|---|
| Simetrica | Nao (KL(P\|\|Q) != KL(Q\|\|P)) | Sim |
| Sempre finita | Nao (infinita se Q=0 em algum bin) | Sim (por usar M como medio) |
| Intervalo | [0, +inf) | [0, 1] |

```python
from scipy.spatial.distance import jensenshannon
# Retorna a RAIZ do JSD (distancia JS), nao o JSD em si
jsd_dist = jensenshannon(p, q)   # em [0, 1]
```

| JSD dist | Status | Interpretacao |
|---|---|---|
| < 0.08 | OK | Distribuicoes similares |
| 0.08 – 0.15 | WARNING | Distancia moderada |
| >= 0.15 | ALERT | Distribuicoes claramente diferentes |

---

### 10. Sistema de Votos para Drift (Exercicio 06)

Usar um unico metodo de deteccao de drift pode gerar falsos positivos (alarme
sem drift real) ou falsos negativos (drift real nao detectado).

A solucao e combinar multiplos metodos com um sistema de votos:

```
Para cada feature:
    voto_ks  = 1 se p_value < 0.05 else 0
    voto_psi = 1 se PSI >= 0.10    else 0
    voto_jsd = 1 se JSD >= 0.08    else 0

    votos = voto_ks + voto_psi + voto_jsd   # em {0, 1, 2, 3}

    severidade = "OK"      se votos == 0
               = "WARNING" se votos == 1
               = "ALERT"   se votos >= 2
```

Isso reduz falsos positivos (exige concordancia de metodos) e aumenta a
confianca na deteccao: quando todos os tres concordam, o drift e real.

---

### 11. Bootstrap como Simulacao de Drift

Para testar monitores de drift, precisamos simular dados com drift controlado.
A estrategia usada nos exercicios e o bootstrap com deslocamento:

```python
# none: reamostra da referencia (PSI proximo de 0 garantido)
base = rng.choice(referencia_samples)
valor = base + 0.0 * std    # sem deslocamento

# low: reamostra + deslocamento leve
valor = base + 0.5 * std

# high: reamostra + deslocamento severo
valor = base + 2.0 * std
```

**Por que bootstrap e nao gaussiana pura?**
Features binarias (como `sex` no Diabetes Dataset) so assumem dois valores
distintos. Amostrar de N(media, std) geraria valores intermediarios que nunca
existiram no dataset real, causando PSI artificialmente alto mesmo sem drift.
O bootstrap preserva a estrutura da distribuicao original.

---

### 12. Metricas de Avaliacao por Tipo de Problema

#### Classificacao

| Metrica | Formula | Quando usar |
|---|---|---|
| **Accuracy** | acertos / total | Classes balanceadas |
| **Precision** | VP / (VP + FP) | Custo alto de falso positivo |
| **Recall** | VP / (VP + FN) | Custo alto de falso negativo |
| **F1** | 2 * P * R / (P + R) | Balanco entre precision e recall |

#### Regressao

| Metrica | Formula | Interpretacao |
|---|---|---|
| **RMSE** | sqrt(mean((y - y_hat)^2)) | Penaliza erros grandes; mesma unidade que y |
| **MAE** | mean(\|y - y_hat\|) | Mais intuitivo; robusto a outliers |
| **R²** | 1 - SS_res/SS_tot | Fracao da variancia explicada (1.0 = perfeito) |

---

## Como Usar Este Repositorio

### Primeira vez

```bash
# Clone o repositorio
git clone https://github.com/lucasvpimentel/intro_mlops.git
cd intro_mlops

# Entre em qualquer exercicio
cd exercicio_01

# Crie o venv e instale dependencias
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Execute o pipeline
python main.py pipeline
```

### Com venv ja criado (uso diario)

```bash
cd exercicio_01
venv\Scripts\activate
python main.py predict 5.1 3.5 1.4 0.2
```

### Dependencias entre exercicios

Os monitores de drift (04, 05, 06) dependem dos dados gerados pelos exercicios
correspondentes (01, 02, 03). Execute na ordem:

```bash
# Ex04 depende do Ex01
cd exercicio_01 && python main.py download && cd ..
cd exercicio_04 && python main.py pipeline

# Ex05 depende do Ex02
cd exercicio_02 && python main.py download && cd ..
cd exercicio_05 && python main.py pipeline

# Ex06 depende do Ex03
cd exercicio_03 && python main.py download && cd ..
cd exercicio_06 && python main.py pipeline
```
