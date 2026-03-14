# 02 — Ecossistema de IA nas Big Clouds

## As tres grandes plataformas

AWS, Azure e Google Cloud oferecem plataformas completas de ML gerenciado.
A ideia e a mesma: abstrair a infraestrutura para que o cientista de dados
foque no modelo, nao em configurar servidores.

---

## Tabela Comparativa — Servicos de ML Gerenciado

| Criterio | AWS SageMaker | Azure Machine Learning | Google Vertex AI |
|---|---|---|---|
| **Lancamento** | 2017 | 2018 | 2021 (unificou AI Platform) |
| **Treinamento** | SageMaker Training Jobs | Azure ML Compute Clusters | Vertex AI Training |
| **Inferencia** | SageMaker Endpoints | Azure ML Managed Endpoints | Vertex AI Prediction |
| **Feature Store** | SageMaker Feature Store | Azure ML Feature Store | Vertex AI Feature Store |
| **AutoML** | SageMaker Autopilot | Azure AutoML | Vertex AutoML |
| **Rastreamento de experimentos** | SageMaker Experiments | Azure ML Experiments | Vertex AI Experiments |
| **Pipelines** | SageMaker Pipelines | Azure ML Pipelines | Vertex AI Pipelines |
| **Marketplace de modelos** | SageMaker JumpStart | Azure AI Model Catalog | Model Garden |
| **Notebooks gerenciados** | SageMaker Studio | Azure ML Studio | Vertex AI Workbench |
| **Monitoramento** | SageMaker Model Monitor | Azure ML Data Drift | Vertex AI Model Monitoring |

---

## AWS SageMaker

O SageMaker e a plataforma mais madura e com mais recursos do mercado.
Fortemente integrada ao ecossistema AWS (S3, Lambda, Step Functions).

### Fluxo tipico no SageMaker

```
S3 (dados) --> SageMaker Studio (notebook) --> Training Job (EC2 com GPU)
           --> Model Registry --> Endpoint (inferencia) --> CloudWatch (logs)
```

### Exemplo: treinar um modelo no SageMaker

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="train.py",       # seu script de treino
    role="arn:aws:iam::...",      # permissoes IAM
    instance_type="ml.m5.xlarge", # tipo de maquina
    framework_version="1.0-1",
)

estimator.fit({"train": "s3://meu-bucket/dados/train.csv"})
# SageMaker cria a instancia, roda o treino, salva o modelo no S3 e destroi a instancia
```

**Pontos fortes:**
- Maior ecossistema de integrações (Step Functions, Lambda, Glue)
- SageMaker JumpStart oferece modelos pre-treinados prontos para deploy
- Feature Store robusto para times grandes

**Pontos fracos:**
- Interface pode ser complexa para iniciantes
- Custo elevado se nao configurado com cuidado
- Vendor lock-in forte

---

## Azure Machine Learning

Forte integracao com o ecossistema Microsoft (Azure DevOps, Power BI,
Active Directory). Popular em empresas que ja usam produtos Microsoft.

### Fluxo tipico no Azure ML

```
Azure Blob Storage --> Azure ML Studio (designer ou SDK) --> Compute Cluster
                   --> Model Registry --> Managed Endpoint --> Application Insights
```

### Exemplo: treinar no Azure ML via SDK

```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment

job = command(
    code="./src",                        # pasta com o codigo
    command="python train.py",
    environment="AzureML-sklearn-1.0",   # ambiente pre-configurado
    compute="gpu-cluster",               # cluster ja criado no portal
    experiment_name="meu-experimento",
)

ml_client.jobs.create_or_update(job)
```

**Pontos fortes:**
- Melhor integracao com Azure DevOps para CI/CD
- Designer visual (drag-and-drop) para montar pipelines sem codigo
- Forte em ambientes corporativos com Active Directory
- Responsible AI dashboard nativo

**Pontos fracos:**
- SDK mudou bastante entre versoes (v1 vs v2)
- Documentacao fragmentada

---

## Google Cloud Vertex AI

A plataforma mais recente das tres, unificou os servicos dispersos do Google
(AI Platform, AutoML, etc.) em 2021. Forte vantagem em modelos de linguagem
e integracao com BigQuery.

### Fluxo tipico no Vertex AI

```
BigQuery / GCS --> Vertex AI Workbench --> Training (custom ou AutoML)
              --> Model Registry --> Vertex AI Prediction --> Cloud Monitoring
```

### Exemplo: treinar no Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project="meu-projeto", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="meu-treino",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
)

job.run(
    dataset=dataset,
    model_display_name="meu-modelo",
    machine_type="n1-standard-4",
)
```

**Pontos fortes:**
- Integracao nativa com BigQuery (dados em escala sem mover arquivos)
- Model Garden com acesso a modelos do Google (Gemini, PaLM)
- Melhor para pipelines de dados + ML em conjunto
- Preco competitivo para treino de longa duracao (TPUs)

**Pontos fracos:**
- Menos maduro que SageMaker em alguns recursos de MLOps
- Ecossistema menor de integrações de terceiros

---

## Como escolher?

```
Ja usa AWS?          --> SageMaker
Ja usa Azure/Office? --> Azure ML
Ja usa BigQuery/GCP? --> Vertex AI
Comecando do zero?   --> Qualquer um; SageMaker tem mais tutoriais
```

Na pratica, muitas empresas grandes usam mais de uma nuvem (multi-cloud)
e ferramentas abertas como MLflow para nao ficarem presas a um vendor.
