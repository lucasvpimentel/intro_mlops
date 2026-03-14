# 07 — Automacao de Pipelines: CI/CD/CT

## O problema sem automacao

```
Sem CI/CD/CT:
  Cientista treina modelo no notebook
  --> manda o .pkl por email para o engenheiro
  --> engenheiro sobe manualmente no servidor
  --> ninguem testa se o novo modelo e melhor
  --> dados novos chegam, modelo fica desatualizado
  --> alguem percebe que o modelo piorou... meses depois
```

Com CI/CD/CT, tudo isso vira codigo e roda automaticamente.

---

## Os tres conceitos

### CI — Continuous Integration (Integracao Continua)

Cada vez que um desenvolvedor faz push de codigo, um pipeline automatico:
1. Roda testes unitarios
2. Verifica qualidade do codigo (linting)
3. Testa se o modelo ainda funciona com os dados atuais

```
git push --> GitHub Actions/GitLab CI --> testes rodam --> aprovado ou reprovado
```

### CD — Continuous Delivery/Deployment (Entrega Continua)

Se o CI passa, o modelo e automaticamente empacotado e disponibilizado
para deploy (Delivery) ou colocado em producao direto (Deployment).

```
CI aprovado --> build do container --> push para registry --> deploy em staging
           --> testes de integracao --> deploy em producao
```

### CT — Continuous Training (Treinamento Continuo)

Especifico de ML. O modelo e retreinado automaticamente quando:
- Novos dados chegam (ex: todo dia a meia-noite)
- Drift e detectado (PSI acima do limiar)
- A metrica de producao cai abaixo do threshold

```
Novos dados chegam --> pipeline de CT dispara --> modelo retreina
                   --> avalia contra modelo atual --> se melhor, promove
```

---

## Implementacao pratica com GitHub Actions

O GitHub Actions permite definir pipelines em arquivos YAML que rodam
automaticamente em eventos (push, pull request, agendamento).

### Exemplo 1: CI para ML

```yaml
# .github/workflows/ci.yml
name: CI - Testes do Modelo

on:
  push:
    branches: [main]
  pull_request:

jobs:
  testes:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Configurar Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Instalar dependencias
        run: pip install -r requirements.txt

      - name: Rodar testes unitarios
        run: python -m pytest tests/ -v

      - name: Verificar qualidade do codigo
        run: |
          pip install flake8
          flake8 src/ --max-line-length=100

      - name: Testar pipeline de treino
        run: python main.py pipeline

      - name: Verificar metrica minima
        run: |
          python -c "
          import json
          with open('data/metrics.json') as f:
              m = json.load(f)
          assert m['accuracy'] >= 0.90, f'Accuracy {m[\"accuracy\"]} abaixo do minimo!'
          print('Metrica OK:', m['accuracy'])
          "
```

### Exemplo 2: CT — retreino agendado

```yaml
# .github/workflows/retrain.yml
name: CT - Retreino Semanal

on:
  schedule:
    - cron: "0 2 * * 1"   # toda segunda-feira as 2h da manha

  workflow_dispatch:       # tambem permite rodar manualmente

jobs:
  retreino:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Instalar dependencias
        run: pip install -r requirements.txt

      - name: Baixar dados mais recentes
        run: python main.py download

      - name: Retreinar modelo
        run: python main.py train

      - name: Avaliar novo modelo
        run: python main.py evaluate

      - name: Comparar com modelo em producao
        run: |
          python scripts/comparar_modelos.py \
            --novo data/metrics_new.json \
            --atual data/metrics_prod.json \
            --threshold 0.02    # aceitar so se melhorar 2%+

      - name: Deploy se aprovado
        run: python scripts/deploy.py

      - name: Notificar equipe
        if: success()
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text": "Retreino concluido! Novo modelo em producao."}'
```

---

## Testes especificos para ML

Em ML, alem dos testes de software tradicionais, precisamos de testes
especificos para o modelo e os dados.

```python
# tests/test_modelo.py
import pytest
import joblib
import pandas as pd
import numpy as np

def test_modelo_carrega():
    """Modelo deve ser carregavel sem erro."""
    model = joblib.load("data/models/iris_model.joblib")
    assert model is not None

def test_predicao_retorna_classe_valida():
    """Predicao deve retornar uma das classes conhecidas."""
    model = joblib.load("data/models/iris_model.joblib")
    X = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                     columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    pred = model.predict(X)
    assert pred[0] in ["setosa", "versicolor", "virginica"]

def test_confianca_entre_0_e_1():
    """Confianca deve estar entre 0 e 1."""
    model = joblib.load("data/models/iris_model.joblib")
    X = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                     columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    proba = model.predict_proba(X)
    assert 0 <= proba.max() <= 1

def test_accuracy_minima():
    """Accuracy no conjunto de teste deve ser >= 90%."""
    import json
    with open("data/metrics.json") as f:
        metrics = json.load(f)
    assert metrics["accuracy"] >= 0.90, f"Accuracy {metrics['accuracy']} abaixo do minimo"

def test_sem_data_leakage():
    """Features nao devem conter o alvo."""
    df = pd.read_csv("data/processed.csv")
    assert "target" not in df.columns or df.columns[-1] == "target"
```

---

## Deploy Gradual — Canary e Blue-Green

Em vez de substituir o modelo antigo de uma vez, fazemos o deploy gradual:

### Canary Deployment
```
100% do trafego --> modelo atual (v1)

Apos deploy do v1:
  5% do trafego  --> modelo novo (v2)   [canary]
  95% do trafego --> modelo atual (v1)

Se metricas do v2 forem boas apos 24h:
  50% --> v2
  50% --> v1

Se tudo certo:
  100% --> v2 (v1 aposentado)
```

### Blue-Green Deployment
```
[Ambiente Blue]  -- modelo atual em producao
[Ambiente Green] -- novo modelo treinado e testado

Switch:
  Load balancer aponta para Green
  Blue fica de standby para rollback imediato
```

O deploy gradual garante que um modelo ruim nao afete todos os usuarios
de uma vez — e permite rollback rapido se algo der errado.
