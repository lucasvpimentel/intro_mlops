"""
src/monitoring/detector.py — Exercicio 05: Drift Monitor (Diabetes)
====================================================================
Responsabilidade: detectar drift usando dois metodos complementares:
    1. Teste KS  — compara distribuicoes inteiras (sensivel a qualquer mudanca)
    2. PSI       — mede deslocamento de populacao por bins (interpretavel)

Por que usar dois metodos?
    O KS test e bom para detectar qualquer diferenca entre distribuicoes,
    mas da uma resposta binaria (drift/nao-drift).

    O PSI e mais interpretavel: seu valor numerico diz O QUANTO a
    distribuicao mudou, nao apenas se mudou. Alem disso, o PSI e o
    metodo mais usado em producao em instituicoes financeiras.

    Juntos, eles dao mais confianca na deteccao.
"""

import numpy as np
from scipy import stats  # para ks_2samp


# Limiares de classificacao do PSI
PSI_OK      = 0.1   # PSI abaixo disso: sem drift
PSI_WARNING = 0.2   # PSI entre 0.1 e 0.2: drift moderado
# PSI acima de 0.2: drift severo


def run_ks_test(reference_samples: list, new_samples: list) -> dict:
    """
    Executa o Teste KS de duas amostras.

    Parametros:
        reference_samples: valores da feature no treino
        new_samples:       valores da feature no novo lote

    Retorna dict com ks_statistic, ks_pvalue e drift_detected.
    """
    ks_stat, p_value = stats.ks_2samp(reference_samples, new_samples)
    return {
        "ks_statistic":   round(float(ks_stat), 4),
        "ks_pvalue":      round(float(p_value), 4),
        "ks_drift":       bool(p_value < 0.05),
    }


def compute_psi(ref_proportions: list, bins: list, new_samples: list) -> dict:
    # Adiciona bordas -inf e +inf para capturar valores fora do range de referencia
    """
    Calcula o PSI (Population Stability Index) usando os bins da referencia.

    A formula do PSI e:
        PSI = sum( (atual_pct - ref_pct) * ln(atual_pct / ref_pct) )

    Parametros:
        ref_proportions: proporcoes por bin da referencia (salvas no reference_stats.json)
        bins:            limites dos bins (breakpoints) da referencia
        new_samples:     valores da feature no novo lote

    Retorna dict com psi_score e psi_status.
    """

    # Recorta amostras ao intervalo da referencia (abordagem padrao em producao).
    # Valores abaixo do minimo vao para o primeiro bin; acima do maximo, para o ultimo.
    # Isso evita instabilidade numerica causada por bins de borda com peso epsilon vs
    # peso real nos dados novos (problema grave em features binarias como 'sex').
    clipped = np.clip(new_samples, bins[0], bins[-1])

    # Conta amostras em cada bin usando os mesmos limites da referencia
    new_counts, _ = np.histogram(clipped, bins=bins)

    # Proporcoes do novo lote com epsilon para evitar log(0)
    new_arr = (new_counts / new_counts.sum()) + 1e-9

    # Proporcoes de referencia (epsilon ja adicionado em prepare_reference.py)
    ref_arr = np.array(ref_proportions)

    min_len = min(len(ref_arr), len(new_arr))
    ref_arr = ref_arr[:min_len]
    new_arr = new_arr[:min_len]

    # Calcula o PSI: soma ponderada das diferencas de log entre atual e referencia
    psi_score = float(np.sum((new_arr - ref_arr) * np.log(new_arr / ref_arr)))

    # Classifica o PSI em tres categorias de severidade
    if psi_score < PSI_OK:
        psi_status = "OK"
    elif psi_score < PSI_WARNING:
        psi_status = "WARNING"
    else:
        psi_status = "ALERT"

    return {
        "psi_score":  round(psi_score, 4),
        "psi_status": psi_status,
        "psi_drift":  bool(psi_score >= PSI_OK),  # qualquer desvio acima do limiar
    }


def detect_drift(ref_stats: dict, new_batch: list, features: list) -> dict:
    """
    Roda KS e PSI para cada feature e agrega os resultados.

    Uma feature e marcada com drift se KS OU PSI indicar drift.
    O status geral e ALERT se qualquer feature tiver drift severo (PSI >= 0.2).
    E WARNING se houver drift moderado. OK caso contrario.
    """

    results = {}

    for feature in features:
        ref_samples     = ref_stats[feature]["samples"]
        ref_proportions = ref_stats[feature]["ref_proportions"]
        bins            = ref_stats[feature]["bins"]
        new_samples     = [row[feature] for row in new_batch]

        # Executa os dois testes
        ks_result  = run_ks_test(ref_samples, new_samples)
        psi_result = compute_psi(ref_proportions, bins, new_samples)

        new_arr = np.array(new_samples)
        results[feature] = {
            **ks_result,
            **psi_result,
            # Drift confirmado se KS E PSI concordam
            "drift_detected":  bool(ks_result["ks_drift"] or psi_result["psi_drift"]),
            "reference_mean":  round(ref_stats[feature]["mean"], 5),
            "new_mean":        round(float(np.mean(new_arr)), 5),
        }

    # Determina status geral com base no PSI mais critico
    max_psi = max(r["psi_score"] for r in results.values())
    if max_psi >= PSI_WARNING:
        overall_status = "ALERT"
    elif max_psi >= PSI_OK:
        overall_status = "WARNING"
    else:
        overall_status = "OK"

    drifted = [f for f, r in results.items() if r["drift_detected"]]

    return {
        "overall_status":      overall_status,
        "drift_detected":      len(drifted) > 0,
        "max_psi":             round(max_psi, 4),
        "features_with_drift": drifted,
        "features_ok":         [f for f in features if f not in drifted],
        "features":            results,
    }
