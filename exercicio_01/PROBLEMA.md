# Exercicio 01 — Classificador de Especies (Iris Dataset)

## Enunciado

O "Hello World" do Machine Learning. O foco aqui e garantir que o modelo
treinado no script possa ser carregado para prever novas flores.

**Objetivo:** Classificar 3 tipos de flores Iris.

## O que fazer em cada pasta

| Pasta | Tarefa |
|---|---|
| `src/data/` | Script para baixar o dataset e salvar como `raw.csv` em `data/` |
| `src/models/` | Treinar uma Regressao Logistica e salvar o arquivo `.joblib` ou `.pkl` |
| Raiz | `predict.py` que aceita 4 medidas via linha de comando e retorna o nome da especie |

## Entrada do modelo

Quatro medidas em centimetros:

1. Comprimento da sepala (`sepal_length`)
2. Largura da sepala (`sepal_width`)
3. Comprimento da petala (`petal_length`)
4. Largura da petala (`petal_width`)

## Saida esperada

Nome da especie prevista (uma das tres classes abaixo) e percentual de confianca:

- `setosa`
- `versicolor`
- `virginica`

## Dataset

- **Nome:** Iris Dataset
- **Origem:** `sklearn.datasets.load_iris`
- **Amostras:** 150 (50 por classe)
- **Features:** 4 numericas
- **Classes:** 3 (balanceadas)
