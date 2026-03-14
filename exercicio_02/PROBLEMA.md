# Exercicio 02 — Estimador de Progressao de Doenca (Diabetes Dataset)

## Enunciado

Um problema de Regressao. O desafio e lidar com a normalizacao das features.

**Objetivo:** Prever um indice quantitativo da progressao da diabetes apos um ano.

## O que fazer em cada pasta

| Pasta | Tarefa |
|---|---|
| `src/data/` | Script para baixar o dataset e salvar como `raw.csv` em `data/` |
| `src/features/` | Criar uma funcao que normalize os dados (StandardScaler) e salve o scaler para ser usado na inferencia |
| `src/models/` | Treinar um modelo de Ridge Regression ou Random Forest Regressor |
| Raiz | `predict.py` que carrega o scaler, normaliza a entrada e retorna o resultado numerico |

## Entrada do modelo

Dez features clinicas (pre-normalizadas pelo dataset original do sklearn):

| Feature | Descricao |
|---|---|
| `age` | Idade |
| `sex` | Sexo |
| `bmi` | Indice de massa corporal |
| `bp` | Pressao arterial media |
| `s1` | Colesterol total (tc) |
| `s2` | LDL |
| `s3` | HDL |
| `s4` | TCH |
| `s5` | Triglicerideos (ltg) |
| `s6` | Glicose |

## Saida esperada

Valor numerico continuo representando o indice de progressao da diabetes
apos um ano.

- Escala: **25** (baixa progressao) a **346** (alta progressao)
- Media do dataset: ~152

## Dataset

- **Nome:** Diabetes Dataset
- **Origem:** `sklearn.datasets.load_diabetes`
- **Amostras:** 442
- **Features:** 10 numericas (pre-normalizadas com media 0)
- **Alvo:** variavel continua (`progression`)

## Desafio principal

As features ja vem com media 0 pelo sklearn, mas o `StandardScaler` deve ser
ajustado nos dados de treino e **persistido** — na inferencia, os dados de
entrada precisam passar pelo mesmo scaler antes de chegar ao modelo.
