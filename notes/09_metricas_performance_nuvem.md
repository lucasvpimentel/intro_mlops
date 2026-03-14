# 09 — Metricas de Performance na Nuvem

## O que medir em producao?

Um modelo pode ter 98% de accuracy e ainda assim ser inutilizavel em producao
se demorar 30 segundos para responder. Metricas de infraestrutura sao tao
importantes quanto metricas de ML.

```
Metricas de ML:         Metricas de Infra:
  accuracy, RMSE, F1      latencia, throughput, disponibilidade
  (quao bom e o modelo)   (quao rapido e confiavel e o sistema)
```

---

## Latencia de Inferencia

**Latencia** e o tempo entre a requisicao chegar e a resposta sair.

### Tipos de latencia

```
t0: requisicao chega no servidor
t1: pre-processamento (normalizar, codificar features)
t2: inferencia (model.predict())
t3: pos-processamento (formatar resposta)
t4: resposta enviada ao cliente

Latencia total = t4 - t0
Latencia do modelo = t3 - t2
```

### Percentis — a metrica certa para latencia

Nunca use a media de latencia. Use percentis:

| Percentil | Significado |
|---|---|
| P50 (mediana) | 50% das requisicoes sao mais rapidas que isso |
| P95 | 95% das requisicoes sao mais rapidas — "caso quase pior" |
| P99 | 99% das requisicoes sao mais rapidas — "pior caso real" |

```
Media de latencia: 120ms   [parece ok]
P50: 80ms                  [metade das requisicoes e rapida]
P95: 450ms                 [5% dos usuarios esperam quase meio segundo]
P99: 2300ms                [1% dos usuarios esperam mais de 2 segundos!]
```

A media esconde os casos ruins. Os usuarios que mais reclamam sao os P99.

### Como medir latencia

```python
import time
import logging

def predict_com_metricas(X, model, scaler):
    inicio = time.perf_counter()

    # Pre-processamento
    t_pre_inicio = time.perf_counter()
    X_scaled = scaler.transform(X)
    t_pre = time.perf_counter() - t_pre_inicio

    # Inferencia
    t_inf_inicio = time.perf_counter()
    pred = model.predict(X_scaled)
    t_inf = time.perf_counter() - t_inf_inicio

    latencia_total = time.perf_counter() - inicio

    # Log para Prometheus/Datadog/CloudWatch
    logging.info({
        "metrica": "latencia",
        "latencia_total_ms":   round(latencia_total * 1000, 2),
        "latencia_preproc_ms": round(t_pre * 1000, 2),
        "latencia_modelo_ms":  round(t_inf * 1000, 2),
    })

    return pred
```

### Targets de latencia por tipo de aplicacao

| Aplicacao | Target P99 |
|---|---|
| Deteccao de fraude em tempo real | < 100ms |
| Recomendacao em e-commerce | < 200ms |
| Chatbot / assistente | < 1000ms |
| Scoring em batch (offline) | sem restricao |

---

## Taxa de Erro

A taxa de erro mede a fracao de requisicoes que falharam (erro 5xx, timeout,
excecao no modelo, etc.).

```
Taxa de erro = requisicoes com erro / total de requisicoes
```

**O que gera erros em modelos de ML:**
- Feature com valor fora do esperado (ex: string onde esperava float)
- Modelo recebe numero diferente de features (schema mudou)
- Timeout por modelo muito lento para o SLA
- Out of memory (modelo muito grande para a instancia)
- Feature ausente que o modelo nao sabe tratar

```python
from prometheus_client import Counter, Histogram

# Metricas Prometheus
requisicoes_total   = Counter("predicoes_total", "Total de requisicoes")
erros_total         = Counter("predicoes_erro", "Total de erros", ["tipo_erro"])
latencia_histograma = Histogram("latencia_segundos", "Latencia das predicoes",
                                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

def predict_seguro(features: dict):
    requisicoes_total.inc()

    try:
        with latencia_histograma.time():
            resultado = fazer_predicao(features)
        return resultado

    except KeyError as e:
        erros_total.labels(tipo_erro="feature_ausente").inc()
        raise ValueError(f"Feature ausente: {e}")

    except Exception as e:
        erros_total.labels(tipo_erro="erro_generico").inc()
        raise
```

**SLA tipico:** taxa de erro < 0.1% (1 erro em 1000 requisicoes).

---

## Throughput

**Throughput** e quantas requisicoes o sistema processa por segundo (RPS).

```
Throughput = requisicoes atendidas / tempo

Se o sistema atende 1000 requisicoes em 10 segundos:
Throughput = 100 RPS
```

### Gargalos de throughput em ML

```
[Requisicao] --> [Load Balancer] --> [API] --> [Pre-proc] --> [Modelo] --> [Resposta]
                                                                  ^
                                              Aqui costuma ser o gargalo
                                              (modelo lento ou instancia pequena)
```

**Como melhorar throughput:**

| Tecnica | Quando usar |
|---|---|
| **Batching** | Agrupa multiplas requisicoes em um unico `model.predict()` |
| **Caching** | Salva resultado de features identicas (ex: Redis) |
| **Model quantization** | Reduz precisao do modelo (float32 -> int8) para acelerar |
| **Horizontal scaling** | Adiciona mais replicas do servidor |
| **GPU** | Para modelos de deep learning, GPU acelera 10-100x vs CPU |

```python
# Exemplo: batching automatico
import asyncio
from collections import deque

fila = deque()

async def predict_com_batch(features):
    """Agrupa requisicoes em batches de ate 32 para eficiencia."""
    future = asyncio.Future()
    fila.append((features, future))

    if len(fila) >= 32:  # batch cheio
        await processar_batch()

    return await future

async def processar_batch():
    batch = [fila.popleft() for _ in range(min(32, len(fila)))]
    features_batch = [item[0] for item in batch]
    futures = [item[1] for item in batch]

    # Uma unica chamada para 32 predicoes
    resultados = model.predict(features_batch)

    for future, resultado in zip(futures, resultados):
        future.set_result(resultado)
```

---

## Eficiencia no Uso de Recursos

### Utilizacao de CPU/GPU

```
CPU ideal: 60-80% de utilizacao
  < 40%: instancia superdimensionada (custo desnecessario)
  > 90%: instancia subdimensionada (latencia aumenta, risco de erros)

GPU ideal: > 70% de utilizacao
  GPU ociosa e dinheiro jogado fora ($1-30/hora por GPU)
```

### Custo por predicao

A metrica de negocio mais importante para infraestrutura:

```
Custo por predicao = custo da instancia por hora / predicoes por hora

Exemplo:
  Instancia ml.m5.xlarge = $0.23/hora
  Throughput = 500 predicoes/segundo = 1.800.000 predicoes/hora
  Custo por predicao = $0.23 / 1.800.000 = $0.000000128 (quase zero)

Mas se o modelo for lento:
  Throughput = 10 predicoes/segundo = 36.000 predicoes/hora
  Custo por predicao = $0.23 / 36.000 = $0.0000064 (50x mais caro)
```

### Dashboard de monitoramento (Grafana)

```
Painel tipico de um modelo em producao:
  [Grafico 1] Latencia P50/P95/P99 ao longo do tempo
  [Grafico 2] Taxa de erro (%) ao longo do tempo
  [Grafico 3] Throughput (RPS) ao longo do tempo
  [Grafico 4] Utilizacao de CPU/GPU
  [Grafico 5] PSI das principais features (drift)
  [Grafico 6] Distribuicao das predicoes (model drift)
```

---

## Alertas de infra — exemplo com thresholds

| Metrica | Warning | Critical | Acao |
|---|---|---|---|
| Latencia P99 | > 500ms | > 2000ms | Escalar horizontalmente |
| Taxa de erro | > 0.5% | > 2% | Investigar logs, rollback |
| CPU | > 80% | > 95% | Escalar ou otimizar |
| GPU | < 30% | - | Reduzir instancia (economia) |
| PSI max | > 0.10 | > 0.20 | Retreino |
