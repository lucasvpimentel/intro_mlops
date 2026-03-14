"""
src/simulation/generate_batch.py — Exercicio 06: Drift Monitor (Wine)
======================================================================
Responsabilidade: simular lotes de vinhos com drift controlado nas 13 features.

Niveis de drift:
    none: sem drift — amostragem da distribuicao de referencia
    low:  drift leve — deslocamento de 0.5 * std
    high: drift alto — deslocamento de 2.0 * std com maior variancia
"""

import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

REF_PATH   = os.path.join(ROOT, "data", "reference_stats.json")
BATCH_PATH = os.path.join(ROOT, "data", "new_batch.json")

FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]
RANDOM_SEED = 42


def generate_batch(drift_level: str = "none", n_samples: int = 60):
    """
    Gera n_samples vinhos simulados e salva em new_batch.json.

    Parametros:
        drift_level (str): nivel de drift ('none', 'low', 'high')
        n_samples   (int): numero de vinhos a gerar (padrao: 60)
    """

    if not os.path.exists(REF_PATH):
        print("reference_stats.json nao encontrado. Execute: python main.py prepare")
        sys.exit(1)

    with open(REF_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)

    # Fator de deslocamento: none=0, low=0.5x std, high=2.0x std
    shift_factor = {"none": 0.0, "low": 0.5, "high": 2.0}[drift_level]

    batch = []
    for _ in range(n_samples):
        sample = {}
        for feature in FEATURES:
            std         = stats[feature]["std"]
            ref_samples = stats[feature]["samples"]

            # Bootstrap da referencia: garante PSI proximo de zero sem drift.
            # Usar Gaussiana causaria PSI alto mesmo sem drift para features
            # com distribuicoes discretas ou assimetricas.
            base_value = float(rng.choice(ref_samples))

            # Deslocamento aditivo proporcional ao std da feature
            value = base_value + shift_factor * std
            sample[feature] = round(value, 4)
        batch.append(sample)

    with open(BATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2)

    print(f"Lote gerado: {n_samples} vinhos com drift='{drift_level}'")
    print(f"Salvo em: {BATCH_PATH}")

    # Resumo comparativo
    print(f"\n{'Feature':<24} {'Ref.Mean':>10} {'Batch Mean':>10} {'Delta':>10}")
    print("-" * 58)
    for feature in FEATURES:
        ref_mean   = stats[feature]["mean"]
        batch_mean = float(np.mean([s[feature] for s in batch]))
        delta      = batch_mean - ref_mean
        print(f"{feature:<24} {ref_mean:>10.3f} {batch_mean:>10.3f} {delta:>+10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", choices=["none", "low", "high"], default="none")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    generate_batch(args.drift, args.n)
