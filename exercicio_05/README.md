# Exercicio 05 — Monitor de Drift com KS + PSI (Diabetes)

Monitor de data drift para o regressor de diabetes (Ex02).
Usa dois metodos complementares: **Teste KS** e **Population Stability Index (PSI)**.

---

## Estrutura do Projeto

```
exercicio_05/
├── data/
│   ├── reference_stats.json   # Estatisticas + bins de histograma do Ex02
│   ├── new_batch.json         # Lote simulado
│   └── drift_report.json      # Relatorio de drift em JSON
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_reference.py   # Calcula stats + bins para PSI
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── detector.py            # KS test + calculo de PSI
│   │   └── report.py              # Relatorio com colunas KS e PSI
│   └── simulation/
│       ├── __init__.py
│       └── generate_batch.py      # Bootstrap + shift para simular drift
├── main.py
└── requirements.txt
```

---

## Pre-requisito

Este exercicio depende dos dados do **Exercicio 02**.

```bash
cd ../exercicio_02
python main.py download
cd ../exercicio_05
```

---

## Configuracao do Ambiente

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

---

## Como Executar

### Passo 1 — Calcular estatisticas de referencia

Alem de media/std/percentis, salva tambem os **bins do histograma** (necessarios para o PSI).

```bash
python main.py prepare
```

### Passo 2 — Simular um novo lote

```bash
python main.py simulate --drift none    # sem drift
python main.py simulate --drift low     # drift leve
python main.py simulate --drift high    # drift severo
python main.py simulate --drift high --n 200
```

### Passo 3 — Detectar drift

```bash
python main.py detect
```

### Pipeline completo

```bash
python main.py pipeline
python main.py pipeline --drift low
python main.py pipeline --drift high
```

---

## Entendendo o Relatorio

```
======================================================================
  RELATORIO DE DRIFT — Diabetes Regressor (Ex02)
======================================================================
  Status geral : OK
  Amostras     : 100
  PSI maximo   : 0.0512
----------------------------------------------------------------------
  Feature   KS-stat    p-val     KS |      PSI   PSI status
  -------- -------- -------- ------   -------- ------------
  age        0.0780   0.7100     ok |   0.0312 +         OK
  bmi        0.0900   0.5300     ok |   0.0290 +         OK
  ...
----------------------------------------------------------------------
  Nenhuma feature com drift detectado.
======================================================================
```

| Coluna     | Significado                                          |
|------------|------------------------------------------------------|
| KS-stat    | Distancia maxima entre CDFs                          |
| p-val      | Probabilidade de diferenca por acaso (< 0.05 = drift)|
| KS         | `DRIFT` ou `ok`                                      |
| PSI        | Score de deslocamento de populacao                   |
| PSI status | `OK` / `WARNING` / `ALERT`; simbolo: `+` / `~` / `!` |

---

## Como o PSI e Calculado

1. Em `prepare_reference.py`, os bins sao criados por percentis uniformes da referencia
2. Em `compute_psi()`, as amostras novas sao recortadas (`clip`) ao range da referencia
3. O histograma novo e comparado ao histograma de referencia bin a bin
4. PSI = sum( (new% - ref%) × ln(new% / ref%) )

### Por que usar clipping?

Valores fora do range de referencia sao atribuidos ao bin mais proximo.
Esta e a abordagem padrao em sistemas de producao, pois evita instabilidade
numerica causada por bins de borda com peso muito pequeno (epsilon) na referencia.

---

## Nota sobre a feature `sex`

O Diabetes Dataset do sklearn normaliza `sex` como variavel continua com apenas
dois valores distintos. O PSI retorna 0 corretamente para `sex` com `drift=none`.
O KS pode sinalizar drift em `sex` mesmo sem deslocamento real, pois e sensivel
a pequenas diferencas de proporcao em distribuicoes discretas — comportamento esperado
e discutido no contexto da disciplina.

---

## Artefatos Salvos e Uso Independente

Apos rodar `prepare` uma vez, as estatisticas de referencia ficam persistidas:

```
data/
├── reference_stats.json   # media, std, percentis, amostras e bins de histograma
├── new_batch.json         # lote simulado mais recente
└── drift_report.json      # relatorio do detect mais recente
```

**Voce nao precisa recalcular a referencia para detectar drift.** Com o `venv`
ja criado e o `reference_stats.json` ja gerado, basta ativar e rodar:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Simular novo lote (ou substituir new_batch.json por dados reais)
python main.py simulate --drift low

# 3. Detectar — so le reference_stats.json e new_batch.json
python main.py detect
```

Em producao, voce substituiria o `new_batch.json` pelos dados reais coletados
no periodo de monitoramento. O `detect` calcula KS + PSI e gera o relatorio
sem retocar a referencia.

---

## Arquitetura e Principios

### Limpeza
`prepare_reference.py` entrega `reference_stats.json` com tudo que o detector precisa.

### Reprodutibilidade
Bootstrap com `np.random.default_rng(42)` garante lotes deterministicos.

### Isolamento
`detector.py` recebe apenas listas/dicionarios Python — sem I/O proprio.

---

## Referencia

- [PSI — Population Stability Index](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations)
- [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
