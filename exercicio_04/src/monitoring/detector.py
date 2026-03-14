"""
src/monitoring/detector.py — Exercicio 04: Drift Monitor (Iris)
================================================================
Responsabilidade: detectar drift comparando um novo lote de dados
contra as estatisticas de referencia usando o Teste KS.

O que e o Teste de Kolmogorov-Smirnov (KS)?
    E um teste estatistico nao-parametrico que compara duas distribuicoes
    sem assumir que elas sejam normais.

    Ele mede a maior diferenca entre as funcoes de distribuicao acumulada
    (CDF) das duas amostras:

        Estatistica KS: D = max|F1(x) - F2(x)|
        p-value: probabilidade de observar essa diferenca por acaso

    Interpretacao:
        p-value < 0.05 → as distribuicoes sao SIGNIFICATIVAMENTE diferentes
                         (drift detectado com 95% de confianca)
        p-value >= 0.05 → nao ha evidencia suficiente de drift
"""

import json
import numpy as np
from scipy import stats  # scipy.stats.ks_2samp: implementacao do teste KS


def run_ks_test(reference_samples: list, new_samples: list) -> dict:
    """
    Executa o Teste KS entre a distribuicao de referencia e o novo lote.

    Parametros:
        reference_samples (list): valores da feature no dataset de treino
        new_samples       (list): valores da mesma feature no novo lote

    Retorna:
        dict com:
            ks_statistic (float): magnitude da diferenca (0=identico, 1=maximo)
            ks_pvalue    (float): p-value do teste (< 0.05 = drift)
            drift_detected (bool): True se p-value < 0.05
    """

    # Executa o teste KS bilateral (two-sided)
    # ks_2samp compara as CDFs das duas amostras
    ks_stat, p_value = stats.ks_2samp(reference_samples, new_samples)

    return {
        "ks_statistic":   round(float(ks_stat), 4),  # D: diferenca maxima entre CDFs
        "ks_pvalue":      round(float(p_value), 4),   # probabilidade de ser acaso
        "drift_detected": bool(p_value < 0.05),        # limiar classico de 5%
    }


def detect_drift(ref_stats: dict, new_batch: list, features: list) -> dict:
    """
    Roda o detector KS para cada feature e agrega os resultados.

    Parametros:
        ref_stats  (dict): estatisticas de referencia (do reference_stats.json)
        new_batch  (list): lista de dicionarios com os dados do novo lote
        features   (list): nomes das features a monitorar

    Retorna:
        dict com resultados por feature e resumo geral
    """

    results = {}  # acumula resultados por feature

    for feature in features:
        # Extrai os valores de referencia (todos do treino)
        ref_samples = ref_stats[feature]["samples"]

        # Extrai os valores do novo lote para esta feature
        new_samples = [row[feature] for row in new_batch]

        # Executa o teste KS
        ks_result = run_ks_test(ref_samples, new_samples)

        # Calcula estatisticas do novo lote para comparacao
        new_arr = np.array(new_samples)
        results[feature] = {
            **ks_result,  # adiciona os resultados do KS (estatistica, p-value, drift_detected)
            "reference_mean": round(ref_stats[feature]["mean"], 4),
            "new_mean":       round(float(np.mean(new_arr)), 4),
            "reference_std":  round(ref_stats[feature]["std"], 4),
            "new_std":        round(float(np.std(new_arr)), 4),
        }

    # Lista de features onde drift foi detectado
    drifted = [f for f, r in results.items() if r["drift_detected"]]

    # Status geral: ALERT se alguma feature tem drift, OK caso contrario
    overall_status = "ALERT" if drifted else "OK"

    return {
        "overall_status":       overall_status,
        "drift_detected":       len(drifted) > 0,
        "features_with_drift":  drifted,
        "features_ok":          [f for f in features if f not in drifted],
        "features":             results,
    }
