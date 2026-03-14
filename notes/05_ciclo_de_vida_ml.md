# 05 — Ciclo de Vida de ML

## Visao geral

O ciclo de vida de um modelo de ML nao termina no treino — ele e continuo.
Um modelo bem construido passa por sete fases, da coleta de dados ao retreino.

```
[1. Coleta] --> [2. EDA] --> [3. Features] --> [4. Treino] -->
[5. Avaliacao] --> [6. Deploy] --> [7. Monitoramento] --> [1. Coleta]
```

---

## Fase 1 — Coleta e Ingestao de Dados

O objetivo e reunir dados brutos de todas as fontes relevantes.

**Fontes tipicas:**
- Bancos de dados relacionais (Postgres, MySQL)
- APIs externas (clima, cotacoes, redes sociais)
- Logs de aplicacao (clickstream, eventos de usuario)
- Arquivos historicos (CSV, Excel legados)
- Sensores IoT (temperatura, vibração, GPS)

**O que fazer nesta fase:**
```python
# Exemplo: coletar dados de uma API e salvar no Data Lake
import requests
import pandas as pd

response = requests.get("https://api.exemplo.com/transacoes?data=2024-03")
dados = pd.DataFrame(response.json())

# Salvar no Data Lake (camada bronze — dado bruto, sem modificar)
dados.to_parquet("s3://meu-bucket/bronze/transacoes/2024-03.parquet")
```

**Armadilhas comuns:**
- Coletar dados sem pensar no label (o que voce quer prever?)
- Viés de seleção: os dados disponiveis nao representam o problema real
- Dados historicos com vazamento do futuro (data leakage)

---

## Fase 2 — Analise Exploratoria (EDA)

Entender o dado antes de modelar. Nenhum algoritmo compensa dados ruins.

**O que inspecionar:**
```python
import pandas as pd
import seaborn as sns

df = pd.read_parquet("dados.parquet")

# Dimensoes e tipos
print(df.shape)          # (10000, 25) -- 10k linhas, 25 colunas
print(df.dtypes)         # tipos de cada coluna
print(df.isnull().sum()) # contagem de nulos por coluna

# Distribuicao dos alvos (classificacao)
df["classe"].value_counts(normalize=True)
# classe_0: 92%   classe_1: 8%  --> dataset desbalanceado!

# Correlacoes
sns.heatmap(df.corr(), annot=True)
```

**Perguntas a responder na EDA:**
1. Ha valores nulos? Em que colunas? Qual o padrao?
2. As classes estao balanceadas?
3. Ha outliers? Sao erros ou casos reais?
4. Quais features se correlacionam com o alvo?
5. Ha features que vazam o futuro (data leakage)?

---

## Fase 3 — Engenharia de Features

Transformar dados brutos em representacoes que o modelo consegue aprender.

**Tipos de transformacao:**

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Tratar valores ausentes
imputer = SimpleImputer(strategy="mean")
X_num = imputer.fit_transform(X[colunas_numericas])

# 2. Normalizar (deixar em escala comparavel)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# 3. Codificar categorias
le = LabelEncoder()
X["cidade_cod"] = le.fit_transform(X["cidade"])

# 4. Criar features novas (feature engineering)
X["idade_ao_quadrado"] = X["idade"] ** 2
X["receita_por_sessao"] = X["receita_total"] / X["total_sessoes"]
X["fim_de_semana"] = X["dia_da_semana"].isin([5, 6]).astype(int)

# REGRA CRITICA: salvar os transformadores para inferencia
import joblib
joblib.dump(scaler, "scaler.joblib")
joblib.dump(imputer, "imputer.joblib")
joblib.dump(le, "le_cidade.joblib")
```

**Feature Importance — descobrir quais features importam:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(10).plot(kind="bar")
```

---

## Fase 4 — Treinamento

Escolher o algoritmo certo e treinar com boas praticas.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Split SEMPRE antes de qualquer transformacao nos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,      # reproducibilidade
    stratify=y,           # manter proporcao das classes
)

# Treinar com validacao cruzada para estimativa confiavel
model = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
print(f"F1 CV: {scores.mean():.3f} +/- {scores.std():.3f}")

# Treinar no conjunto completo de treino
model.fit(X_train, y_train)

# Salvar o modelo
import joblib
joblib.dump(model, "modelo.joblib")
```

**Armadilhas do treino:**
- Aplicar o scaler no conjunto de teste antes de separar (data leakage)
- Nao usar seed (resultados irreprodutivies)
- Otimizar a metrica errada (accuracy em dados desbalanceados)

---

## Fase 5 — Avaliacao

Medir o desempenho no conjunto de teste (dados nunca vistos no treino).

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

# Relatorio completo
print(classification_report(y_test, y_pred))

# Matriz de confusao
cm = confusion_matrix(y_test, y_pred)
#          Predito: 0   1
# Real: 0 [  TN    FP ]
# Real: 1 [  FN    TP ]

# Para regressao
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} | R2: {r2:.4f}")
```

**Criterio de go/no-go:**
Defina antes de treinar qual metrica e qual valor minimo aceita.
Nao mude os criterios depois de ver o resultado.

---

## Fase 6 — Deploy

Colocar o modelo em producao para receber dados reais.

**Opcoes de deploy:**

| Modalidade | Descricao | Exemplo |
|---|---|---|
| **Batch** | Roda em horario fixo, processa lote | Scoring noturno de credito |
| **REST API** | Endpoint HTTP que responde em tempo real | App de recomendacao |
| **Streaming** | Processa eventos continuos | Deteccao de fraude em tempo real |
| **Embedded** | Modelo no proprio dispositivo | App mobile com modelo local |

```python
# Exemplo: servir o modelo como API com FastAPI
from fastapi import FastAPI
import joblib, pandas as pd

app = FastAPI()
model  = joblib.load("modelo.joblib")
scaler = joblib.load("scaler.joblib")

@app.post("/predict")
def predict(dados: dict):
    X = pd.DataFrame([dados])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0].max()
    return {"predicao": int(pred), "confianca": round(float(proba), 3)}
```

---

## Fase 7 — Monitoramento e Retreino

O modelo em producao precisa ser observado continuamente.

```
Dados de producao --> Monitorar features (data drift)
                 --> Monitorar predicoes (model drift)
                 --> Monitorar latencia e erros (infra)
                 --> Se drift detectado --> disparar retreino
```

**Quando retreinar?**
- Data drift acima do limiar (PSI > 0.2, JSD > 0.15)
- Metrica de negocio caiu abaixo do threshold (ex: taxa de aprovacao incomum)
- Calendario fixo (ex: toda segunda-feira com dados da semana)
- Mudanca conhecida no ambiente (novo produto, nova regiao, pandemia)

O ciclo entao recomeça: novos dados chegam, o pipeline e disparado,
um novo modelo e treinado, avaliado e, se melhor, promovido a producao.
