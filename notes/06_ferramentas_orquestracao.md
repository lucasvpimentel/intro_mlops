# 06 — Ferramentas e Orquestracao

## Por que precisamos de orquestracao?

Um pipeline de ML em producao tem muitos passos:
ingestao → limpeza → features → treino → avaliacao → deploy → monitoramento.

Cada passo pode falhar. Passos dependem uns dos outros. Precisamos de
reproducibilidade, logs e a capacidade de re-rodar apenas o passo que falhou.

**Orquestracao** e o que gerencia essa sequencia de forma confiavel.

---

## Kubeflow — ML sobre Kubernetes

O **Kubeflow** e uma plataforma open-source que roda pipelines de ML
em cima do Kubernetes (o sistema de orquestracao de containers do Google).

### Por que Kubernetes para ML?

```
Sem Kubernetes:
  "O treino rodou na minha maquina com Python 3.9, numpy 1.23, CUDA 11.8"
  "No servidor tem Python 3.11, numpy 2.0... deu erro"

Com Kubernetes + Docker:
  O pipeline roda em um container identico em qualquer lugar
```

### Como um pipeline Kubeflow funciona

Cada passo e um **componente** — uma funcao Python empacotada em container.

```python
from kfp import dsl, compiler

# Componente 1: preparar dados
@dsl.component(base_image="python:3.10")
def preparar_dados(caminho_entrada: str, caminho_saida: str):
    import pandas as pd
    df = pd.read_csv(caminho_entrada)
    df = df.dropna()
    df.to_parquet(caminho_saida)

# Componente 2: treinar modelo
@dsl.component(base_image="python:3.10", packages_to_install=["scikit-learn"])
def treinar(caminho_dados: str, caminho_modelo: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_parquet(caminho_dados)
    X, y = df.drop("alvo", axis=1), df["alvo"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, caminho_modelo)

# Pipeline: conecta os componentes
@dsl.pipeline(name="pipeline-iris")
def meu_pipeline():
    passo1 = preparar_dados(
        caminho_entrada="gs://bucket/raw.csv",
        caminho_saida="gs://bucket/clean.parquet",
    )
    passo2 = treinar(
        caminho_dados=passo1.output,
        caminho_modelo="gs://bucket/modelo.joblib",
    )

# Compilar e submeter
compiler.Compiler().compile(meu_pipeline, "pipeline.yaml")
```

**O que o Kubeflow oferece:**
- Interface visual para acompanhar cada passo do pipeline
- Re-execucao a partir do passo que falhou
- Cache de resultados (nao re-executa passo com mesmo input)
- Historico de todas as execucoes

---

## MLflow — Rastreamento de Experimentos

O **MLflow** e a ferramenta mais usada para registrar e comparar experimentos.
Sem ele, e facil perder qual combinacao de hiperparametros gerou o melhor modelo.

### O problema que o MLflow resolve

```
Experimento 1: RandomForest, n_estimators=100 --> accuracy 0.87
Experimento 2: RandomForest, n_estimators=200 --> accuracy 0.91
Experimento 3: GradientBoosting, lr=0.01     --> accuracy 0.93
Experimento 4: GradientBoosting, lr=0.001    --> accuracy ???

"Qual foi o melhor? Quais eram os parametros exatos?"
```

### Como usar o MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("classificador-iris")

# Cada "run" e um experimento registrado
with mlflow.start_run():
    # 1. Registrar parametros
    n_estimators = 200
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", 42)

    # 2. Treinar
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # 3. Registrar metricas
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 4. Salvar o modelo versionado
    mlflow.sklearn.log_model(model, "modelo")
    # Artefatos extras
    mlflow.log_artifact("confusion_matrix.png")
```

**MLflow UI — comparar experimentos:**
```bash
mlflow ui
# Abre em http://localhost:5000
# Voce ve uma tabela com todos os runs, parametros e metricas
# Pode ordenar por accuracy, filtrar por parametro, etc.
```

### Model Registry — versionamento de modelos

```python
# Promover um modelo para "Staging" e depois "Production"
client = mlflow.MlflowClient()

# Registrar
client.create_registered_model("ClassificadorIris")
client.create_model_version(
    name="ClassificadorIris",
    source="runs:/abc123/modelo",
    run_id="abc123",
)

# Transicionar de Staging para Production
client.transition_model_version_stage(
    name="ClassificadorIris",
    version=3,
    stage="Production",
)

# Carregar o modelo de producao (sem saber qual e o arquivo)
model = mlflow.sklearn.load_model("models:/ClassificadorIris/Production")
```

---

## TFX — Pipelines de Producao do Google

O **TensorFlow Extended (TFX)** e o framework do Google para pipelines de ML
em producao. Mais opinativo que Kubeflow — cada componente tem uma responsabilidade clara.

### Componentes do TFX

```
ExampleGen  --> le e divide os dados (treino/avaliacao)
    |
StatisticsGen --> calcula estatisticas descritivas
    |
SchemaGen   --> infere o schema esperado dos dados
    |
ExampleValidator --> detecta anomalias nos dados
    |
Transform   --> feature engineering (salva o grafo de transformacao)
    |
Trainer     --> treina o modelo (compativel com TF, sklearn, etc.)
    |
Evaluator   --> avalia e compara com o modelo em producao
    |
Pusher      --> faz deploy se o novo modelo for melhor
```

**Diferencial do TFX:** o componente `Transform` salva o grafo de
pre-processamento junto com o modelo. Isso garante que treino e producao
aplicam exatamente a mesma transformacao — eliminando training-serving skew.

---

## Comparativo das ferramentas

| Ferramenta | Foco principal | Melhor para |
|---|---|---|
| **Kubeflow** | Orquestracao de pipelines em Kubernetes | Times que ja usam K8s |
| **MLflow** | Rastreamento de experimentos e registry | Todos os times (facil de adotar) |
| **TFX** | Pipeline completo opinativo para producao | Times com TensorFlow |
| **Airflow** | Agendamento de tarefas (nao especifico de ML) | Orquestracao geral de dados |
| **Prefect/Dagster** | Alternativas modernas ao Airflow | Pipelines de dados + ML |

### Combinacao mais comum na industria

```
MLflow (rastreamento) + Kubeflow (orquestracao) + S3/GCS (armazenamento)
ou
MLflow (rastreamento) + Airflow (agendamento) + SageMaker/Vertex (infra)
```

O MLflow e quase universal — a maioria dos times o usa independente
de qual orquestrador escolhe para os pipelines.
