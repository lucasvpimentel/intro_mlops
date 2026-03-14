"""
src/simulation/generate_batch.py — Exercicio 05: Drift Monitor (Diabetes)
==========================================================================
Responsabilidade: simular novos lotes de pacientes com drift controlado.

O Diabetes Dataset ja vem pre-normalizado pelo sklearn (media ~0, std ~0.05).
Os deslocamentos de drift sao proporcionais ao desvio padrao real de cada feature.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REF_PATH   = ROOT / "data" / "reference_stats.json"
BATCH_PATH = ROOT / "data" / "new_batch.json"

FEATURES    = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
RANDOM_SEED = 42


def generate_batch(drift_level: str = "none", n_samples: int = 100):
    """
    Gera n_samples pacientes simulados e salva em new_batch.json.

    Niveis de drift:
        none: amostragem direta da distribuicao de referencia
        low:  deslocamento de 0.5 * std em todas as features
        high: deslocamento de 2.0 * std + aumento de variancia

    Parametros:
        drift_level (str): 'none', 'low' ou 'high'
        n_samples   (int): numero de pacientes a gerar
    """

    if not REF_PATH.exists():
        print("reference_stats.json nao encontrado. Execute: python main.py prepare")
        sys.exit(1)

    with open(REF_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)

    # Fator de deslocamento proporcional ao desvio padrao de cada feature
    # none: 0 (sem drift), low: 0.5x std, high: 2.0x std
    shift_factor = {"none": 0.0, "low": 0.5, "high": 2.0}[drift_level]

    batch = []
    for _ in range(n_samples):
        sample = {}
        for feature in FEATURES:
            std           = stats[feature]["std"]
            ref_samples   = stats[feature]["samples"]

            # Reamostragem (bootstrap) da referencia garante que features binarias
            # (como 'sex') mantenham a distribuicao original com drift='none'.
            # Isso e mais correto do que gerar valores de uma Gaussiana, pois o PSI
            # fica proximo de zero quando nao ha drift real.
            base_value = float(rng.choice(ref_samples))

            # Adiciona deslocamento proporcional ao desvio padrao para simular drift
            shift = shift_factor * std

            value = base_value + shift
            sample[feature] = round(value, 6)
        batch.append(sample)

    with open(BATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2)

    print(f"Lote gerado: {n_samples} pacientes com drift='{drift_level}'")
    print(f"Salvo em: {BATCH_PATH}")

    # Exibe comparacao de medias
    print(f"\n{'Feature':<8} {'Ref.Mean':>10} {'Batch Mean':>10} {'Delta':>10}")
    print("-" * 42)
    for feature in FEATURES:
        ref_mean   = stats[feature]["mean"]
        batch_mean = float(np.mean([s[feature] for s in batch]))
        delta      = batch_mean - ref_mean
        print(f"{feature:<8} {ref_mean:>10.5f} {batch_mean:>10.5f} {delta:>+10.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", choices=["none", "low", "high"], default="none")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    generate_batch(args.drift, args.n)
