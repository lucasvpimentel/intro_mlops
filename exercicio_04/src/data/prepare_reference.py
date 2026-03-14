"""
src/data/prepare_reference.py — Exercicio 04: Drift Monitor (Iris)
====================================================================
Responsabilidade: calcular e salvar as estatisticas de referencia
a partir dos dados de TREINO do Exercicio 01.

O que sao estatisticas de referencia?
    Sao os "valores esperados" de cada feature quando o modelo foi treinado.
    Em producao, comparamos os dados novos contra essas estatisticas para
    detectar se a distribuicao mudou (drift).

    Exemplo: se 'petal_length' tinha media 3.7 no treino e agora esta
    chegando com media 5.5, algo mudou — pode ser sazonalidade, erro de
    sensor ou mudanca real na populacao.

Por que salvar as estatisticas e nao o dataset inteiro?
    Em producao, o dataset original pode ter milhoes de linhas.
    Salvar apenas as estatisticas (media, desvio, percentis, distribuicao
    amostrada) e mais eficiente e suficiente para os testes de drift.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Caminho raiz deste exercicio
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Aponta para os dados de treino do Exercicio 01
EX01_DATA = os.path.normpath(os.path.join(ROOT, "..", "exercicio_01", "data", "raw.csv"))

# Onde salvar as estatisticas de referencia
REF_PATH = os.path.join(ROOT, "data", "reference_stats.json")

# As 4 features continuas do Iris Dataset
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def prepare_reference():
    """
    Le o dataset de treino do Ex01, calcula estatisticas descritivas
    por feature e salva em data/reference_stats.json.

    Estatisticas salvas por feature:
        mean:    media aritmetica
        std:     desvio padrao
        min/max: valores extremos
        q25/q50/q75: quartis (25%, 50%, 75%)
        samples: lista com todos os valores (para KS test posterior)

    Nao recebe parametros e nao retorna nada.
    """

    # Verifica se os dados do Ex01 existem
    if not os.path.exists(EX01_DATA):
        print(f"Dados do Ex01 nao encontrados em: {EX01_DATA}")
        print("Execute primeiro: cd ../exercicio_01 && python main.py download")
        sys.exit(1)

    # Carrega o dataset de treino original
    df = pd.read_csv(EX01_DATA)
    print(f"Dados de referencia carregados: {df.shape[0]} amostras, {len(FEATURES)} features")

    # Dicionario que acumulara as estatisticas de cada feature
    stats = {}

    for feature in FEATURES:
        # Pega todos os valores da coluna como array numpy
        values = df[feature].values

        # Calcula as estatisticas descritivas
        stats[feature] = {
            "mean":    float(np.mean(values)),       # media: centro da distribuicao
            "std":     float(np.std(values)),         # desvio padrao: espalhamento
            "min":     float(np.min(values)),         # valor minimo observado
            "max":     float(np.max(values)),         # valor maximo observado
            "q25":     float(np.percentile(values, 25)),  # 1o quartil
            "q50":     float(np.percentile(values, 50)),  # mediana
            "q75":     float(np.percentile(values, 75)),  # 3o quartil
            # Salva todos os valores para uso no KS test (teste nao-parametrico
            # que compara as distribuicoes inteiras, nao apenas estatisticas resumidas)
            "samples": values.tolist(),
        }

        print(f"  {feature}: mean={stats[feature]['mean']:.3f}, std={stats[feature]['std']:.3f}")

    # Salva o dicionario de estatisticas como arquivo JSON
    with open(REF_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)  # indent=2 deixa o JSON legivel

    print(f"\nEstatisticas de referencia salvas em: {REF_PATH}")


if __name__ == "__main__":
    prepare_reference()
