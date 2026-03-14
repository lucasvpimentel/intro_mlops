# Exercicio 04 — Monitor de Drift com Teste KS (Iris)

## Contexto

Voce treinou um classificador de especies de Iris no Exercicio 01 e o colocou em producao.
Apos algumas semanas, a equipe de campo percebe que as predicoes comecaram a deteriorar.
O suspeito: os dados de entrada mudaram — a distribuicao das medidas das flores coletadas
nesta nova temporada e diferente da distribuicao usada no treino.

Sua tarefa e construir um **monitor de drift** que detecte automaticamente quando isso acontece.

## Objetivo

Implementar um sistema de monitoramento que:

1. Calcule estatisticas de referencia a partir dos dados de treino do Exercicio 01.
2. Simule novos lotes de dados com diferentes niveis de drift controlado.
3. Execute o **Teste de Kolmogorov-Smirnov (KS)** para cada feature numerica.
4. Gere um relatorio textual e salve o resultado em JSON.

## O que e Drift?

**Data drift** (ou covariate shift) ocorre quando a distribuicao dos dados de entrada
muda apos o modelo ser treinado. Isso pode degradar a qualidade das predicoes sem que
nenhum erro explicito seja lancado — o modelo continua funcionando, mas com resultados piores.

## O que e o Teste KS?

O **Teste de Kolmogorov-Smirnov de duas amostras** compara as funcoes de distribuicao
acumulada (CDF) de dois conjuntos de dados. Ele responde: "estas duas amostras vieram
da mesma distribuicao?"

- **Estatistica KS**: maxima diferenca absoluta entre as duas CDFs (0 = identicas, 1 = maxima diferenca)
- **p-value**: probabilidade de observar essa diferenca por acaso. Se p < 0.05, o drift e estatisticamente significativo.

## Niveis de Drift Simulados

| Nivel | Deslocamento | Descricao                              |
|-------|-------------|----------------------------------------|
| none  | 0.0 × std   | Sem drift — dados identicos a referencia |
| low   | 0.5 × std   | Drift leve — distribuicao levemente deslocada |
| high  | 2.0 × std   | Drift severo — distribuicao claramente diferente |

## Features Monitoradas

As 4 features numericas do Iris Dataset:
- `sepal_length` — comprimento da sepala (cm)
- `sepal_width`  — largura da sepala (cm)
- `petal_length` — comprimento da petala (cm)
- `petal_width`  — largura da petala (cm)

## Resultado Esperado

- **drift=none**: todas as features com p-value alto (> 0.05), nenhum drift detectado
- **drift=high**: todas as features com p-value baixo (< 0.05), drift detectado em todas

## Dependencia

Este exercicio usa os dados gerados pelo **Exercicio 01**.
Execute `python main.py download` no exercicio_01 antes de iniciar este exercicio.
