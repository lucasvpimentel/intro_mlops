# 08 — Monitoramento em Producao

## Por que monitorar?

Um modelo treinado hoje pode ser excelente. Daqui a seis meses,
pode estar errado na maioria das vezes — sem nenhum erro de codigo.

Os dados do mundo mudam. Comportamentos de usuarios mudam. Politicas mudam.
Sem monitoramento, o modelo se degrada silenciosamente.

```
Janeiro:  modelo com 94% de accuracy
Marco:    accuracy cai para 88% -- ninguem percebeu
Junho:    accuracy esta em 71%  -- negocio comeca a sentir
Agosto:   equipe descobre -- 8 meses de modelo ruim em producao
```

---

## Os dois tipos principais de drift

### Data Drift (Covariate Shift)

A distribuicao dos dados de entrada (features X) mudou, mas a relacao
entre X e Y continua a mesma.

```
Treino:    clientes de 25-40 anos, renda media R$5k
Producao:  clientes de 40-60 anos, renda media R$12k

O modelo nunca viu esse perfil. As predicoes ficam imprecisas.
```

**Como detectar:** comparar a distribuicao das features de producao
com a distribuicao do treino usando KS Test, PSI ou JSD.
(Exatamente o que os exercicios 04, 05 e 06 implementam.)

### Model Drift (Concept Drift)

A relacao entre X e Y mudou. O mesmo perfil de cliente agora se comporta
diferente do que no passado.

```
Treino (pre-pandemia):  pessoa com renda alta --> baixo risco de inadimplencia
Producao (pandemia):    pessoa com renda alta --> risco variavel (demissoes)

A relacao mudou. O modelo precisa ser retreinado com dados novos.
```

**Como detectar:** monitorar a metrica de negocio ao longo do tempo
(accuracy, precision, recall, RMSE) ou comparar predicoes com os
resultados reais quando eles ficam disponiveis.

---

## Arquitetura de monitoramento

```
[Modelo em Producao]
        |
        v
[Logger de Predicoes] --> banco de dados de predicoes
        |                  (feature + predicao + timestamp)
        v
[Job de Monitoramento] -- roda periodicamente (ex: diario)
        |
        |--> Monitorar features: KS / PSI / JSD
        |--> Monitorar predicoes: distribuicao das saidas
        |--> Monitorar performance: se labels reais disponiveis
        |--> Monitorar infra: latencia, taxa de erro, uso de CPU/GPU
        |
        v
[Alertas] --> Slack / email / PagerDuty
        |
        v
[Retreino] --> dispara CT pipeline se drift acima do limiar
```

---

## Monitoramento de features (Data Drift)

```python
import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

def monitorar_drift(reference_stats: dict, new_batch: list, features: list):
    """
    Compara a distribuicao atual com a referencia de treino.
    Retorna um relatorio por feature com KS, PSI e JSD.
    """
    alertas = []

    for feature in features:
        ref_samples = reference_stats[feature]["samples"]
        new_samples = [row[feature] for row in new_batch]

        # KS Test
        ks_stat, p_value = stats.ks_2samp(ref_samples, new_samples)

        if p_value < 0.05:
            alertas.append({
                "feature": feature,
                "metodo": "KS",
                "severity": "ALERT",
                "detalhe": f"p-value={p_value:.4f}",
            })
            print(f"DRIFT DETECTADO em {feature}: KS p={p_value:.4f}")

    return alertas

# Rodar diariamente
alertas = monitorar_drift(reference_stats, dados_de_hoje, FEATURES)

if alertas:
    enviar_alerta_slack(alertas)
    if drift_severo(alertas):
        disparar_retreino()
```

---

## Logging de predicoes

Para detectar model drift, precisamos armazenar cada predicao feita
e, quando o resultado real ficar disponivel, comparar.

```python
import logging
import json
from datetime import datetime

# Configurar logger estruturado
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modelo")

def predict_com_log(features: dict, model, scaler) -> dict:
    """
    Faz predicao e loga tudo para monitoramento posterior.
    """
    import pandas as pd

    X = pd.DataFrame([features])
    X_scaled = scaler.transform(X)

    predicao = int(model.predict(X_scaled)[0])
    confianca = float(model.predict_proba(X_scaled)[0].max())

    # Log estruturado -- vai para CloudWatch, Datadog, etc.
    logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": gerar_id_unico(),
        "features": features,
        "predicao": predicao,
        "confianca": confianca,
        "modelo_versao": "v3.2",
    }))

    return {"predicao": predicao, "confianca": confianca}
```

**Por que log estruturado (JSON)?**
Ferramentas como CloudWatch Insights, Splunk e Datadog conseguem
indexar e fazer queries em JSON automaticamente:

```sql
-- Query no CloudWatch Insights para ver taxa de confianca baixa
fields @timestamp, features.idade, predicao, confianca
| filter confianca < 0.6
| stats count() as alertas by bin(1h)
| sort @timestamp desc
```

---

## Ferramentas de observabilidade para ML

| Ferramenta | O que monitora | Tipo |
|---|---|---|
| **Evidently AI** | Data drift, model performance | Open-source |
| **WhyLabs** | Drift e anomalias em producao | SaaS |
| **Arize AI** | Performance e drift com embeddings | SaaS |
| **AWS SageMaker Monitor** | Drift e qualidade integrado ao SageMaker | AWS |
| **Vertex AI Monitoring** | Drift e skew automatico | GCP |
| **Prometheus + Grafana** | Metricas de infra (latencia, erros) | Open-source |
| **Datadog** | Logs, metricas e APM (paid) | SaaS |

### Exemplo com Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Comparar dados de referencia com dados de producao
report = Report(metrics=[
    DataDriftPreset(),        # detecta drift em todas as features
    ClassificationPreset(),   # metricas de classificacao se tiver labels
])

report.run(
    reference_data=df_treino,
    current_data=df_producao_ultimo_mes,
)

report.save_html("relatorio_drift.html")  # abre no browser com graficos
```

---

## Politica de alertas — quando agir?

Nem todo drift exige retreino imediato. Defina niveis:

| Nivel | Criterio | Acao |
|---|---|---|
| **INFO** | PSI 0.05-0.10 | Registrar, nenhuma acao |
| **WARNING** | PSI 0.10-0.20 ou 1 metodo detecta drift | Investigar, agendar analise |
| **ALERT** | PSI > 0.20 ou 2+ metodos detectam drift | Retreino agendado para proxima janela |
| **CRITICO** | Metrica de negocio cai > 5% | Retreino imediato + notificacao ao gestor |

A definicao dos limiares depende do negocio: um modelo de deteccao de
fraude precisa de alertas mais sensiveis que um modelo de recomendacao.
