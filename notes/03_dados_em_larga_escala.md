# 03 — Dados em Larga Escala

## O problema

Modelos de ML sao tao bons quanto os dados que recebem. Em producao,
os dados chegam de multiplas fontes, em formatos diferentes, em volumes
que nao cabem em memoria. Precisamos de uma estrategia de armazenamento
e processamento que escale.

---

## Data Lake — O repositorio central

Um **Data Lake** e um repositorio centralizado que armazena dados brutos
em qualquer formato (CSV, JSON, Parquet, imagens, audio) sem estrutura pre-definida.

```
Fontes de dados --> Data Lake --> Processamento --> Modelo ML
   (raw, bruto)      (S3, GCS)    (Athena, BigQuery)
```

### Camadas do Data Lake (arquitetura Medallion)

```
Bronze (raw)    -- dado bruto, exatamente como chegou
    |
    v
Silver (limpo)  -- dado tratado, sem nulos criticos, schema consistente
    |
    v
Gold (pronto)   -- features prontas para treinar o modelo
```

**Exemplo pratico:**
```
Bronze: logs de clickstream brutos (JSON, 500GB/dia)
Silver: eventos parseados, timestamps normalizados, bots removidos
Gold:   features agregadas por usuario (media de sessao, produtos vistos)
```

### Por que nao usar um banco de dados relacional?

| Criterio | Banco SQL (Postgres) | Data Lake (S3 + Parquet) |
|---|---|---|
| Volume maximo pratico | ~TBs | Ilimitado (PBs) |
| Custo de armazenamento | Alto | Muito baixo (~$0.023/GB no S3) |
| Schema | Rigido (definido antes) | Flexivel (schema-on-read) |
| Velocidade de escrita | Alta | Media |
| Velocidade de leitura analítica | Media | Alta (com Parquet + particionamento) |

---

## Parquet — O formato certo para ML

Parquet e um formato de arquivo **colunar** e comprimido. Em vez de guardar
linha por linha (como CSV), guarda coluna por coluna.

```
CSV (linha a linha):
  [id, nome, idade, compras]
  [1, Ana, 30, 5]
  [2, Joao, 25, 12]

Parquet (coluna a coluna):
  [id]:      [1, 2, ...]
  [nome]:    [Ana, Joao, ...]
  [idade]:   [30, 25, ...]
  [compras]: [5, 12, ...]
```

**Por que isso importa para ML?**
Se voce quer calcular a media de `compras`, o Parquet le so a coluna
`compras` — ignora `nome`, `idade`, etc. O CSV leria tudo.

```python
# Salvar features como Parquet
import pandas as pd

df.to_parquet("features.parquet", compression="snappy")

# Ler so as colunas que o modelo precisa
X = pd.read_parquet("features.parquet", columns=["idade", "compras", "regiao"])
```

---

## Amazon Athena — SQL direto no S3

O Athena e um servico da AWS que permite rodar queries SQL em arquivos
diretamente no S3, sem precisar carregar os dados em um banco.

```
S3 (arquivo Parquet) --> Athena (query SQL) --> resultado em segundos
(voce paga por TB lido, ~$5/TB)
```

### Exemplo pratico

```sql
-- Athena: calcular features de usuarios para treino
-- O dado fica no S3, Athena le e processa distribuido

SELECT
    user_id,
    COUNT(*)                    AS total_sessoes,
    AVG(duracao_segundos)       AS media_duracao,
    SUM(valor_compra)           AS receita_total,
    MAX(data_evento)            AS ultima_atividade
FROM "s3://meu-data-lake/eventos/ano=2024/mes=03/"
WHERE tipo_evento = 'compra'
GROUP BY user_id;
```

**Vantagens:**
- Zero infraestrutura para gerenciar (serverless)
- Paga so pelo que le (incentiva usar Parquet para reduzir custo)
- Particionamento por pasta reduz o volume lido

**Particionamento no S3:**
```
s3://meu-bucket/eventos/
  ano=2024/mes=01/dia=01/parte_0.parquet
  ano=2024/mes=01/dia=02/parte_0.parquet
  ano=2024/mes=02/...

-- Query so no mes 01: Athena le apenas as pastas ano=2024/mes=01/
-- Sem particionamento: Athena leria TUDO
```

---

## BigQuery — Data Warehouse do Google para ML

O BigQuery e o equivalente do Google: um data warehouse serverless que
escala automaticamente e tem integracao nativa com o Vertex AI.

```
Fontes --> BigQuery --> Vertex AI Feature Store --> Modelo
                    --> Vertex AI Training (direto do BQ)
```

### Exemplo: treinar um modelo direto do BigQuery

```python
from google.cloud import bigquery
from google.cloud import aiplatform

# Criar dataset de treino diretamente de uma query no BigQuery
dataset = aiplatform.TabularDataset.create(
    display_name="dataset-fraude",
    bq_source="bq://meu-projeto.meu-dataset.tabela_features",
)

# Treinar AutoML usando esse dataset
job = aiplatform.AutoMLTabularTrainingJob(
    display_name="modelo-fraude",
    optimization_prediction_type="classification",
)
job.run(dataset=dataset, target_column="is_fraude")
```

**Diferenciais do BigQuery para ML:**
- `BigQuery ML`: treina modelos SQL diretamente no BQ sem exportar dados
- Integracao direta com Vertex AI (sem mover arquivos)
- Particao e clustering automaticos

### BigQuery ML — treinar sem sair do SQL

```sql
-- Treinar uma regressao logistica direto no BigQuery
CREATE OR REPLACE MODEL `meu_projeto.dataset.modelo_fraude`
OPTIONS(
    model_type = 'logistic_reg',
    input_label_cols = ['is_fraude']
) AS
SELECT
    valor_transacao,
    hora_do_dia,
    pais_origem,
    is_fraude
FROM `meu_projeto.dataset.transacoes`
WHERE data BETWEEN '2024-01-01' AND '2024-12-31';

-- Avaliar
SELECT * FROM ML.EVALUATE(MODEL `meu_projeto.dataset.modelo_fraude`);

-- Prever
SELECT * FROM ML.PREDICT(MODEL `meu_projeto.dataset.modelo_fraude`,
    TABLE `meu_projeto.dataset.novas_transacoes`);
```

---

## Estrategia completa de dados para ML em producao

```
[Ingestao]
  APIs, logs, bancos --> Kafka/Pub-Sub --> Data Lake Bronze (S3/GCS)

[Processamento]
  Spark / dbt --> Silver (dados limpos) --> Gold (features prontas)

[Servico de Features]
  Feature Store (SageMaker / Vertex / Feast) --> mesmo dado no treino e inferencia

[Treino]
  Gold layer --> Athena/BigQuery query --> DataFrame --> modelo treinado

[Inferencia]
  Feature Store --> modelo --> predicao em tempo real
```

O **Feature Store** e o componente critico: garante que as features usadas
no treino sao identicas as usadas na inferencia em producao — eliminando
o **training-serving skew** (diferenca de calculo entre treino e producao).
