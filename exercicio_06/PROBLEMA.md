# Exercicio 06 — Monitor de Drift com KS + PSI + JSD (Wine)

## Contexto

O classificador de vinhos do Exercicio 03 e usado por uma vinicultura para
controle de qualidade. Apos uma safra atipica com condicoes climaticas diferentes,
o sommelier chefe nota que as classificacoes automaticas nao batem com sua avaliacao.
Suspeita-se que as caracteristicas quimicas dos vinhos dessa safra mudaram
em relacao ao historico de treino.

Sua tarefa e construir o monitor de drift mais completo da serie, usando **tres metodos simultaneos**.

## Objetivo

Implementar um sistema de monitoramento que:

1. Calcule estatisticas de referencia das 13 features quimicas do Wine Dataset.
2. Simule novos lotes com drift controlado em tres niveis.
3. Execute **KS Test**, **PSI** e **Jensen-Shannon Divergence (JSD)**.
4. Classifique cada feature por severidade usando sistema de votos (0/1/2+ metodos).
5. Gere relatorio completo com tabela multi-metodo.

## Os Tres Metodos

### 1. Teste KS (Kolmogorov-Smirnov)
- Compara as CDFs empiricas das duas amostras
- Resposta: drift detectado (p < 0.05) ou nao
- Ponto forte: rigoroso, nao-parametrico, sensivel a qualquer diferenca

### 2. PSI (Population Stability Index)
- Mede deslocamento de populacao usando histogramas fixos
- Classifica: OK (< 0.1) | WARNING (0.1–0.2) | ALERT (>= 0.2)
- Ponto forte: numerico e interpretavel, padrao em producao financeira

### 3. JSD (Jensen-Shannon Divergence)
- Distancia simetrica entre distribuicoes de probabilidade
- Baseada na divergencia KL mas sempre finita e simetrica
- Classifica: OK (< 0.08) | WARNING (0.08–0.15) | ALERT (>= 0.15)
- Ponto forte: simetrica, robusta, intervalo bem definido [0, 1]

**Diferenca entre KL e JS:**
```
KL(P||Q) = sum( P * log(P/Q) )   — assimetrica, pode ser infinita
JS(P, Q) = (KL(P||M) + KL(Q||M)) / 2  onde M = (P+Q)/2   — simetrica, [0, 1]
```
A funcao `scipy.spatial.distance.jensenshannon` retorna a **raiz quadrada** do JSD
(a distancia JS), que e uma metrica no sentido matematico estrito.

## Sistema de Votos

Cada metodo vota: detectou drift (1) ou nao (0).

| Votos | Severidade | Interpretacao         |
|-------|------------|----------------------|
| 0     | OK         | Nenhum sinal de drift |
| 1     | WARNING    | Um metodo sinaliza — monitorar |
| 2+    | ALERT      | Consenso de drift — investigar |

## Features Monitoradas

As 13 caracteristicas quimicas do Wine Dataset:
`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`,
`total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`,
`color_intensity`, `hue`, `od280_od315`, `proline`

## Niveis de Drift Simulados

| Nivel | Deslocamento | PSI esperado | JSD esperado |
|-------|-------------|-------------|-------------|
| none  | 0.0 × std   | < 0.15 (ruido amostral) | < 0.12 |
| low   | 0.5 × std   | > 0.3 | > 0.20 |
| high  | 2.0 × std   | > 8.0 | > 0.50 |

**Nota sobre amostras pequenas**: O Wine Dataset tem apenas 178 amostras de referencia.
Lotes de 120 amostras (bootstrap) tem variancia amostral significativa, podendo
gerar alguns falsos positivos com `drift=none`. Isso e esperado e serve como
exemplo didatico das limitacoes de monitores com pequenos volumes de dados.

## Dependencia

Este exercicio usa os dados do **Exercicio 03**.
Execute `python main.py download` no exercicio_03 antes de iniciar este exercicio.
