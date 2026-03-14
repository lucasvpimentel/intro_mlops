# Exercicio 04 — Monitor de Drift com Teste KS (Iris)

Monitor de data drift para o classificador de Iris (Ex01).
Detecta mudancas de distribuicao nas features usando o **Teste de Kolmogorov-Smirnov**.

---

## Estrutura do Projeto

```
exercicio_04/
├── data/
│   ├── reference_stats.json   # Estatisticas calculadas do Ex01
│   ├── new_batch.json         # Lote simulado para deteccao
│   └── drift_report.json      # Relatorio gerado pelo detector
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_reference.py   # Le dados do Ex01, calcula estatisticas
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── detector.py            # Executa o Teste KS por feature
│   │   └── report.py              # Imprime relatorio e salva JSON
│   └── simulation/
│       ├── __init__.py
│       └── generate_batch.py      # Simula novos lotes com drift controlado
├── main.py
└── requirements.txt
```

---

## Pre-requisito

Este exercicio depende dos dados do **Exercicio 01**. Certifique-se de que o arquivo
`../exercicio_01/data/raw.csv` existe antes de prosseguir.

```bash
cd ../exercicio_01
python main.py download
cd ../exercicio_04
```

---

## Configuracao do Ambiente

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Como Executar

### Passo 1 — Calcular estatisticas de referencia

Le os dados do Ex01 e salva media, desvio padrao, percentis e amostras por feature.

```bash
python main.py prepare
```

### Passo 2 — Simular um novo lote

Gera um conjunto de amostras com drift controlado.

```bash
python main.py simulate --drift none    # sem drift
python main.py simulate --drift low     # drift leve (0.5 * std)
python main.py simulate --drift high    # drift severo (2.0 * std)
python main.py simulate --drift high --n 200  # 200 amostras
```

### Passo 3 — Detectar drift

Executa o Teste KS para cada feature e gera o relatorio.

```bash
python main.py detect
```

### Pipeline completo (todos os passos de uma vez)

```bash
python main.py pipeline                      # sem drift
python main.py pipeline --drift low          # drift leve
python main.py pipeline --drift high         # drift severo
```

---

## Entendendo o Relatorio

```
======================================================================
  RELATORIO DE DRIFT — Iris Classifier (Ex01)
======================================================================
  Status geral : OK
  Amostras     : 50
----------------------------------------------------------------------
  Feature      KS-stat   p-value   Drift
  ---------- --------- --------- -------
  sepal_len     0.1200    0.8200      no
  sepal_wid     0.1100    0.8700      no
  petal_len     0.1000    0.9200      no
  petal_wid     0.0900    0.9500      no
----------------------------------------------------------------------
  Nenhuma feature com drift detectado.
======================================================================
```

| Coluna    | Significado                                              |
|-----------|----------------------------------------------------------|
| KS-stat   | Maxima diferenca entre as CDFs (0=identicas, 1=maximas) |
| p-value   | Probabilidade de diferenca por acaso (< 0.05 = drift)   |
| Drift     | `DRIFT` se p < 0.05, `ok` caso contrario                |

---

## Artefatos Salvos e Uso Independente

Apos rodar `prepare` uma vez, as estatisticas de referencia ficam persistidas:

```
data/
├── reference_stats.json   # estatisticas do Ex01 (media, std, percentis, amostras)
├── new_batch.json         # lote simulado mais recente
└── drift_report.json      # relatorio do detect mais recente
```

**Voce nao precisa recalcular a referencia para detectar drift.** Com o `venv`
ja criado e o `reference_stats.json` ja gerado, basta ativar, simular um lote
e detectar:

```bash
# 1. Ativar o venv existente (nao precisa recriar)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Gerar novo lote (ou trazer um arquivo new_batch.json real do campo)
python main.py simulate --drift high

# 3. Detectar drift — so le os JSONs, nao recalcula referencia
python main.py detect
```

Em producao real, o `new_batch.json` seria substituido pelos dados coletados
recentemente (formato: lista de dicionarios com as 4 features).
O `detect` le os dois JSONs e gera o relatorio sem nenhum retreino.

---

## Arquitetura e Principios

### Limpeza
`prepare_reference.py` entrega as estatisticas prontas para o detector — sem processamento adicional.

### Reprodutibilidade
O gerador de lotes usa `np.random.default_rng(42)` e bootstrap da referencia.
O mesmo comando sempre produz o mesmo arquivo `new_batch.json`.

### Isolamento
`detector.py` so recebe dicionarios Python. Nao le arquivos diretamente.
`report.py` so exibe e persiste — nao calcula nada.

---

## Referencia

- [Kolmogorov-Smirnov Test — Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
