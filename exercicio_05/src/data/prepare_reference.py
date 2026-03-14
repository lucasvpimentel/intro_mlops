"""
src/data/prepare_reference.py — Exercicio 05: Drift Monitor (Diabetes)
=======================================================================
Responsabilidade: calcular e salvar as estatisticas de referencia
a partir dos dados de treino do Exercicio 02.

Diferenca em relacao ao Ex04:
    Aqui tambem salvamos os bins de histograma de cada feature, necessarios
    para o calculo do PSI (Population Stability Index) no detector.

O que e PSI?
    Metrica muito usada em modelos de credito e risco financeiro.
    Mede o quanto a distribuicao atual desviou da referencia:

        PSI = sum( (atual% - esperado%) * ln(atual% / esperado%) )

    Interpretacao:
        PSI < 0.1  : sem drift significativo
        0.1 <= PSI < 0.2 : drift moderado — monitorar
        PSI >= 0.2 : drift severo — investigar e possivelmente retreinar
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Aponta para os dados brutos do Exercicio 02
EX02_DATA = ROOT.parent / "exercicio_02" / "data" / "raw.csv"

REF_PATH = ROOT / "data" / "reference_stats.json"

# As 10 features clinicas do Diabetes Dataset
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# Numero de bins para o histograma usado no PSI
N_BINS = 10


def prepare_reference():
    """
    Calcula estatisticas de referencia das 10 features do Diabetes Dataset
    e salva em data/reference_stats.json.

    Alem das estatisticas descritivas (media, std, percentis), salva
    tambem os bins e as proporcoes do histograma para uso no PSI.

    Nao recebe parametros e nao retorna nada.
    """

    if not EX02_DATA.exists():
        print(f"Dados do Ex02 nao encontrados em: {EX02_DATA}")
        print("Execute: cd ../exercicio_02 && python main.py download")
        sys.exit(1)

    df = pd.read_csv(EX02_DATA)
    print(f"Referencia carregada: {df.shape[0]} amostras, {len(FEATURES)} features")

    stats = {}
    for feature in FEATURES:
        values = df[feature].values

        # Calcula os limites dos bins usando percentis uniformes da referencia
        # np.linspace(0, 100, N_BINS + 1) gera [0, 10, 20, ..., 100]
        # Usar percentis garante que cada bin tenha amostras representativas
        breakpoints = np.percentile(values, np.linspace(0, 100, N_BINS + 1))

        # np.unique remove duplicatas nos breakpoints (ocorre quando ha muitos valores iguais)
        breakpoints = np.unique(breakpoints)

        # Conta quantas amostras caem em cada bin
        counts, _ = np.histogram(values, bins=breakpoints)

        # Converte contagens para proporcoes (percentual por bin)
        # Adiciona epsilon (1e-9) para evitar divisao por zero no calculo do PSI
        proportions = (counts / counts.sum()) + 1e-9

        stats[feature] = {
            "mean":        float(np.mean(values)),
            "std":         float(np.std(values)),
            "min":         float(np.min(values)),
            "max":         float(np.max(values)),
            "q25":         float(np.percentile(values, 25)),
            "q50":         float(np.percentile(values, 50)),
            "q75":         float(np.percentile(values, 75)),
            "samples":     values.tolist(),         # para o KS test
            "bins":        breakpoints.tolist(),    # limites dos bins para o PSI
            "ref_proportions": proportions.tolist(), # proporcoes de referencia para o PSI
        }
        print(f"  {feature}: mean={stats[feature]['mean']:.4f}, std={stats[feature]['std']:.4f}")

    with open(REF_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEstatisticas de referencia salvas em: {REF_PATH}")


if __name__ == "__main__":
    prepare_reference()
