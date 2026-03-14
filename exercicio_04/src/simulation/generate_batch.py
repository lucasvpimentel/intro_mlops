"""
src/simulation/generate_batch.py — Exercicio 04: Drift Monitor (Iris)
======================================================================
Responsabilidade: simular um novo lote (batch) de dados de entrada,
com tres niveis possiveis de drift.

O que e simulacao de drift?
    Em producao, os dados novos chegam continuamente. Para testar o
    monitor, simulamos tres cenarios:

    none: sem drift  — dados gerados da mesma distribuicao do treino
                       (amostragem direta do dataset original)

    low:  drift leve — media de cada feature deslocada em 0.5 desvios padrao
                       (mudanca sutil, pode passar despercebida)

    high: drift alto — media deslocada em 2.0 desvios padrao + ruido extra
                       (mudanca clara, o monitor deve alertar)

Uso:
    python src/simulation/generate_batch.py [--drift none|low|high]
"""

import os
import sys
import json
import argparse
import numpy as np

# Caminho raiz do exercicio
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Estatisticas de referencia geradas pelo prepare_reference.py
REF_PATH   = os.path.join(ROOT, "data", "reference_stats.json")

# Arquivo de saida: o lote simulado que o detector vai analisar
BATCH_PATH = os.path.join(ROOT, "data", "new_batch.json")

# Features do Iris Dataset
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Semente aleatoria para reproducibilidade
RANDOM_SEED = 42


def generate_batch(drift_level: str = "none", n_samples: int = 50):
    """
    Gera um lote de n_samples flores simuladas e salva em new_batch.json.

    Parametros:
        drift_level (str): nivel de drift a simular
                           'none' = sem drift
                           'low'  = drift leve (0.5 * std de deslocamento)
                           'high' = drift alto (2.0 * std de deslocamento)
        n_samples   (int): numero de amostras a gerar (padrao: 50)

    Nao retorna nada — salva o arquivo JSON em disco.
    """

    # Verifica se as estatisticas de referencia existem
    if not os.path.exists(REF_PATH):
        print("reference_stats.json nao encontrado.")
        print("Execute: python main.py prepare")
        sys.exit(1)

    # Carrega as estatisticas de referencia
    with open(REF_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Inicializa o gerador de numeros aleatorios com semente fixa
    rng = np.random.default_rng(RANDOM_SEED)

    # Define o fator de deslocamento (shift) com base no nivel de drift
    # Multiplo do desvio padrao que sera adicionado a media
    shift_factor = {
        "none": 0.0,   # sem deslocamento
        "low":  0.5,   # metade do desvio padrao
        "high": 2.0,   # dois desvios padrao
    }[drift_level]

    # Lista que acumula as amostras geradas
    batch = []

    for _ in range(n_samples):
        # Para cada amostra, gera um valor por feature
        sample = {}

        for feature in FEATURES:
            mean = stats[feature]["mean"]   # media de referencia
            std  = stats[feature]["std"]    # desvio padrao de referencia

            # Valor base: sorteado de uma distribuicao normal com os parametros
            # de referencia (sem drift isso ja e suficiente)
            base_value = rng.normal(loc=mean, scale=std)

            # Aplica o deslocamento proporcional ao desvio padrao
            # shift_factor=0.0 para 'none', 0.5 para 'low', 2.0 para 'high'
            drift_shift = shift_factor * std

            # Valor final: base + deslocamento
            sample[feature] = round(float(base_value + drift_shift), 4)

        batch.append(sample)

    # Salva o lote como JSON
    with open(BATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2)

    print(f"Lote gerado: {n_samples} amostras com drift='{drift_level}'")
    print(f"Salvo em: {BATCH_PATH}")

    # Exibe um resumo das medias do lote gerado vs referencia
    print("\nComparacao de medias (referencia vs lote gerado):")
    print(f"{'Feature':<20} {'Ref. Mean':>12} {'Batch Mean':>12} {'Diferenca':>12}")
    print("-" * 58)
    batch_df_values = {f: [s[f] for s in batch] for f in FEATURES}
    for feature in FEATURES:
        ref_mean   = stats[feature]["mean"]
        batch_mean = float(np.mean(batch_df_values[feature]))
        diff       = batch_mean - ref_mean
        print(f"{feature:<20} {ref_mean:>12.4f} {batch_mean:>12.4f} {diff:>+12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drift", choices=["none", "low", "high"], default="none",
        help="Nivel de drift a simular (padrao: none)"
    )
    parser.add_argument("--n", type=int, default=50, help="Numero de amostras")
    args = parser.parse_args()
    generate_batch(args.drift, args.n)
