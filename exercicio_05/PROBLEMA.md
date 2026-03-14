# Exercicio 05 — Monitor de Drift com KS + PSI (Diabetes)

## Contexto

O modelo de regressao do Exercicio 02 estima a progressao da diabetes em pacientes.
Em producao, o hospital observou que as estimativas de risco ficaram sistematicamente
subestimadas em um grupo de pacientes. Suspeita-se que a distribuicao das variaveis
clinicas (IMC, pressao sanguinea, colesterol) mudou em relacao ao periodo de treino.

Sua tarefa e construir um monitor de drift mais robusto que use **dois metodos complementares**.

## Objetivo

Implementar um sistema de monitoramento que:

1. Calcule estatisticas de referencia (media, std, percentis, bins de histograma) do Ex02.
2. Simule novos lotes de pacientes com drift controlado.
3. Execute o **Teste KS** (detecta qualquer diferenca de distribuicao).
4. Calcule o **PSI** (Population Stability Index — quantifica o deslocamento).
5. Gere relatorio combinando os dois metodos.

## Por que dois metodos?

| Metodo | Ponto Forte | Limitacao |
|--------|------------|-----------|
| KS Test | Detecta qualquer diferenca distribucional; rigoroso estatisticamente | Resposta binaria (drift/nao-drift) |
| PSI | Numerico e interpretavel; indica O QUANTO mudou; padrao em credit scoring | Requer binagem; sensivel ao numero de bins |

Juntos, os dois metodos reduzem falsos positivos e oferecem mais contexto.

## O que e PSI?

O **Population Stability Index** mede o deslocamento entre a distribuicao de referencia
e a distribuicao atual usando bins de histograma:

```
PSI = sum( (atual% - esperado%) × ln(atual% / esperado%) )
```

### Interpretacao

| PSI | Status  | Acao recomendada              |
|-----|---------|-------------------------------|
| < 0.1 | OK      | Modelo estavel               |
| 0.1 – 0.2 | WARNING | Monitorar de perto           |
| >= 0.2 | ALERT   | Investigar e possivelmente retreinar |

## Features Monitoradas

As 10 features clinicas (ja normalizadas pelo sklearn):
`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`

**Nota tecnica**: `sex` e uma variavel binaria normalizada. O KS pode sinalizar
drift nessa feature mesmo com amostras aleatorias, pois e sensivel a pequenas
diferencas de proporcao em variaveis discretas. O PSI, com binagem por clipping,
corretamente retorna 0 quando nao ha deslocamento real.

## Niveis de Drift Simulados

| Nivel | Deslocamento | Acao simulada                     |
|-------|-------------|-----------------------------------|
| none  | 0.0 × std   | Bootstrap direto da referencia    |
| low   | 0.5 × std   | Bootstrap + deslocamento leve     |
| high  | 2.0 × std   | Bootstrap + deslocamento severo   |

## Dependencia

Este exercicio usa os dados do **Exercicio 02**.
Execute `python main.py download` no exercicio_02 antes de iniciar este exercicio.
