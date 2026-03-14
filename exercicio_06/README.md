# Exercicio 06 — Monitor de Drift com KS + PSI + JSD (Wine)

Monitor de data drift para o classificador de vinhos (Ex03).
Tres metodos simultaneos: **KS Test**, **PSI** e **Jensen-Shannon Divergence**.
Classificacao de severidade por sistema de votos (0 → OK | 1 → WARNING | 2+ → ALERT).

---

## Estrutura do Projeto

```
exercicio_06/
├── data/
│   ├── reference_stats.json   # Stats + bins de histograma do Ex03
│   ├── new_batch.json         # Lote simulado
│   └── drift_report.json      # Relatorio de drift em JSON
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_reference.py   # Calcula stats + bins para PSI e JSD
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── detector.py            # KS + PSI + JSD com sistema de votos
│   │   └── report.py              # Relatorio wide com todos os metodos
│   └── simulation/
│       ├── __init__.py
│       └── generate_batch.py      # Bootstrap + shift nas 13 features
├── main.py
└── requirements.txt
```

---

## Pre-requisito

Este exercicio depende dos dados do **Exercicio 03**.

```bash
cd ../exercicio_03
python main.py download
cd ../exercicio_06
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

```bash
python main.py prepare
```

### Passo 2 — Simular um novo lote

```bash
python main.py simulate --drift none           # sem drift (120 amostras)
python main.py simulate --drift low            # drift leve
python main.py simulate --drift high           # drift severo
python main.py simulate --drift high --n 200   # 200 amostras
```

### Passo 3 — Detectar drift

```bash
python main.py detect
```

### Pipeline completo

```bash
python main.py pipeline                  # none drift
python main.py pipeline --drift low      # low drift
python main.py pipeline --drift high     # high drift
```

---

## Entendendo o Relatorio

```
================================================================================
  RELATORIO DE DRIFT — Wine Classifier (Ex03)
================================================================================
  Status geral  : [OK]
  Amostras      : 120
--------------------------------------------------------------------------------
  Feature                  KS-stat  p-val    PSI  PSI-st    JSD  JSD-st Votos    Sev
  ------------------------ ------- ------ ------ ------- ------ ------- ----- ------
  alcohol                   0.0648 0.8995 0.1127 WARNING 0.1178 WARNING     2   ALRT
  magnesium                 0.0393 0.9996 0.0250      OK 0.0559      OK     0    OK
  ...
================================================================================
  Legenda votos: numero de metodos (KS/PSI/JSD) que detectaram drift
  Severidade: OK = 0 votos | WARNING = 1 voto | ALERT = 2+ votos
================================================================================
```

| Coluna   | Significado                                                    |
|----------|----------------------------------------------------------------|
| KS-stat  | Distancia maxima entre CDFs                                    |
| p-val    | p-value do KS (< 0.05 = drift detectado)                      |
| PSI      | Score de deslocamento de populacao (bins fixos da referencia)  |
| PSI-st   | Status do PSI: OK / WARNING / ALERT                            |
| JSD      | Jensen-Shannon Distance (raiz do JSD, em [0, 1])              |
| JSD-st   | Status do JSD: OK / WARNING / ALERT                            |
| Votos    | Quantos dos 3 metodos detectaram drift                         |
| Sev      | Severidade: OK / WARN / ALRT                                   |

---

## Limiares dos Metodos

| Metodo | OK          | WARNING       | ALERT      |
|--------|-------------|---------------|------------|
| KS     | p >= 0.05   | —             | p < 0.05   |
| PSI    | < 0.10      | 0.10 – 0.20   | >= 0.20    |
| JSD    | < 0.08      | 0.08 – 0.15   | >= 0.15    |

> Os limiares do JSD foram calibrados para lotes de ~120 amostras.
> Em producao com volumes maiores, valores menores (ex: 0.05 / 0.10) sao mais comuns.

---

## Nota sobre Amostras Pequenas

O Wine Dataset tem apenas 178 amostras de referencia.
Com lotes de 120 (bootstrap com reposicao), a variancia amostral e significativa.
Com `drift=none`, e esperado que alguns metodos sinalizem WARNING em algumas features —
isso e ruido amostral, nao drift real. Com `drift=high`, todos os metodos concordam
claramente com ALERT em todas as features.

Este comportamento e didatico: ilustra a diferenca entre **ruido amostral** e **drift real**,
e a importancia de calibrar limiares de acordo com o volume de dados disponivel.

---

## Artefatos Salvos e Uso Independente

Apos rodar `prepare` uma vez, as estatisticas de referencia ficam persistidas:

```
data/
├── reference_stats.json   # media, std, amostras e bins das 13 features do Wine
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
python main.py simulate --drift high --n 120

# 3. Detectar — so le reference_stats.json e new_batch.json
python main.py detect
```

Em producao, voce substituiria o `new_batch.json` pelos dados quimicos da nova
safra. O `detect` calcula KS + PSI + JSD e gera o relatorio completo com o
sistema de votos — sem retocar a referencia.

---

## Arquitetura e Principios

### Limpeza
`prepare_reference.py` centraliza todos os calculos de referencia em um unico arquivo JSON.

### Reprodutibilidade
Bootstrap com `np.random.default_rng(42)` garante lotes deterministicos e reproziveis.

### Isolamento
`detector.py` so opera sobre estruturas Python em memoria — sem I/O proprio.

---

## Referencia

- [Jensen-Shannon Divergence — Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
- [scipy.spatial.distance.jensenshannon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html)
- [PSI em Credit Scoring](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations)
