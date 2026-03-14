# Exercicio 03 — Classificacao de Qualidade de Vinhos (Wine Dataset)

## Enunciado

Este envolve mais variaveis e a necessidade de separar bem o que e
processamento do que e modelo.

**Objetivo:** Classificar vinhos em 3 classes de cultivo diferentes.

## O que fazer em cada pasta

| Pasta | Tarefa |
|---|---|
| `src/data/` | Script para baixar o dataset e salvar como `raw.csv` |
| `src/features/` | Normalizar com `StandardScaler` e salvar o scaler |
| `src/models/` | Treinar um classificador, avaliar separadamente e fazer inferencia em lote |
| `notebooks/` | Analise de correlacao entre os componentes quimicos do vinho |
| `scripts/` | `run_pipeline.bat` que executa ingestao, treino e avaliacao em sequencia |

## Inferencia em lote (batch inference)

Ao inves de um script que aceita argumentos via linha de comando,
a inferencia le um arquivo `data/input.json` com dados de N vinhos
e gera `data/output.csv` com as previsoes.

## Dataset

- **Nome:** Wine Dataset
- **Origem:** `sklearn.datasets.load_wine`
- **Amostras:** 178
- **Features:** 13 componentes quimicos
- **Classes:** 3 cultivares (`class_0`, `class_1`, `class_2`)

## Features disponíveis

| Feature | Descricao |
|---|---|
| `alcohol` | Teor alcoolico |
| `malic_acid` | Acido malico |
| `ash` | Cinzas |
| `alcalinity_of_ash` | Alcalinidade das cinzas |
| `magnesium` | Magnesio |
| `total_phenols` | Fenois totais |
| `flavanoids` | Flavanoides |
| `nonflavanoid_phenols` | Fenois nao-flavanoides |
| `proanthocyanins` | Proantocianinas |
| `color_intensity` | Intensidade de cor |
| `hue` | Matiz |
| `od280_od315` | OD280/OD315 de vinhos diluidos |
| `proline` | Prolina |

## Desafio principal

Com 13 features e correlacoes entre elas (multicolinearidade), e importante:
- Entender quais features sao mais discriminativas (notebook)
- Separar preprocessamento (features/) de modelagem (models/)
- Garantir que o scaler do treino e reutilizado na inferencia em lote
