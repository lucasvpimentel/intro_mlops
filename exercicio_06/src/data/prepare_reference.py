"""
src/data/prepare_reference.py — Exercicio 06: Drift Monitor (Wine)
===================================================================
Responsabilidade: calcular estatisticas de referencia do Wine Dataset (Ex03),
com suporte a tres metodos de deteccao: KS, PSI e Jensen-Shannon Divergence.

O que e Jensen-Shannon Divergence (JSD)?
    E uma versao simetrica e suavizada da KL Divergence.
    Mede a "distancia" entre duas distribuicoes de probabilidade.

    Vantagens sobre KL Divergence:
        - E simetrica: JSD(P||Q) = JSD(Q||P)
        - Sempre finita (mesmo quando uma distribuicao tem zeros)
        - Raiz quadrada do JSD e uma metrica de distancia valida

    Escala: 0 (identicas) a 1 (completamente diferentes)

    Limiares praticos:
        JSD < 0.05: distribuicoes muito proximas
        0.05 <= JSD < 0.1: leve divergencia
        JSD >= 0.1: divergencia significativa
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

EX03_DATA = ROOT.parent / "exercicio_03" / "data" / "raw.csv"
REF_PATH  = ROOT / "data" / "reference_stats.json"

FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]

N_BINS = 10  # numero de bins para histograma (PSI e JSD)


def prepare_reference():
    """
    Le o dataset do Ex03 e salva estatisticas de referencia com bins
    para KS, PSI e Jensen-Shannon Divergence.

    As proporcoes do histograma sao salvas normalizadas (somam 1.0),
    prontas para uso direto como distribuicao de probabilidade no JSD.

    Nao recebe parametros e nao retorna nada.
    """

    if not EX03_DATA.exists():
        print(f"Dados do Ex03 nao encontrados em: {EX03_DATA}")
        print("Execute: cd ../exercicio_03 && python main.py download")
        sys.exit(1)

    df = pd.read_csv(EX03_DATA)
    print(f"Referencia carregada: {df.shape[0]} amostras, {len(FEATURES)} features")

    stats = {}
    for feature in FEATURES:
        values = df[feature].values

        # Breakpoints baseados em percentis uniformes da referencia
        breakpoints = np.percentile(values, np.linspace(0, 100, N_BINS + 1))
        breakpoints = np.unique(breakpoints)

        # Histograma normalizado: proporcoes que somam 1.0
        # Usado tanto para PSI quanto para JSD (que requer distribuicao de probabilidade)
        counts, _   = np.histogram(values, bins=breakpoints)
        proportions = (counts / counts.sum()) + 1e-9  # epsilon evita log(0) no JSD

        stats[feature] = {
            "mean":            float(np.mean(values)),
            "std":             float(np.std(values)),
            "min":             float(np.min(values)),
            "max":             float(np.max(values)),
            "q25":             float(np.percentile(values, 25)),
            "q50":             float(np.percentile(values, 50)),
            "q75":             float(np.percentile(values, 75)),
            "samples":         values.tolist(),          # para KS test
            "bins":            breakpoints.tolist(),     # breakpoints compartilhados
            "ref_proportions": proportions.tolist(),     # para PSI e JSD
        }
        print(f"  {feature:<22}: mean={stats[feature]['mean']:.3f}")

    with open(REF_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEstatisticas de referencia salvas em: {REF_PATH}")


if __name__ == "__main__":
    prepare_reference()
