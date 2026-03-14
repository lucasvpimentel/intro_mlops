"""
src/monitoring/detector.py — Exercicio 06: Drift Monitor (Wine)
================================================================
Responsabilidade: detectar drift com tres metodos complementares:

    1. KS Test           — compara CDFs das distribuicoes (nao-parametrico)
    2. PSI               — Population Stability Index (interpretavel, por bins)
    3. JS Divergence     — Jensen-Shannon Divergence (metrica simetrica de distancia)

A combinacao dos tres metodos reduz falsos positivos e falsos negativos:
    - KS e sensivel a qualquer diferenca na distribuicao
    - PSI e interpretavel e amplamente usado em producao
    - JSD e simetrico e sempre finito (robusto a bins vazios)

Severidade final por feature:
    OK      — nenhum metodo detectou drift
    WARNING — pelo menos 1 metodo detectou drift moderado
    ALERT   — 2 ou mais metodos detectaram drift
"""

import numpy as np
from scipy import stats                     # ks_2samp
from scipy.spatial.distance import jensenshannon  # JSD


# Limiares de cada metodo
KS_ALPHA    = 0.05   # p-value do KS abaixo disso = drift
PSI_WARN    = 0.1    # PSI entre 0.1-0.2 = moderado
PSI_ALERT   = 0.2    # PSI >= 0.2 = severo
JSD_WARN    = 0.08   # JSD entre 0.08-0.15 = moderado  (calibrado para lotes de ~120 amostras)
JSD_ALERT   = 0.15   # JSD >= 0.15 = severo


def run_ks_test(reference_samples: list, new_samples: list) -> dict:
    """
    Teste KS de duas amostras — detecta qualquer diferenca entre distribuicoes.

    Retorna ks_statistic, ks_pvalue e ks_drift (bool).
    """
    ks_stat, p_val = stats.ks_2samp(reference_samples, new_samples)
    return {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue":    round(float(p_val), 4),
        "ks_drift":     bool(p_val < KS_ALPHA),
    }


def compute_psi(ref_proportions: list, bins: list, new_samples: list) -> dict:
    """
    PSI: mede deslocamento de populacao usando bins fixos da referencia.

    Usa bins estendidos com -inf/+inf para capturar valores fora do range
    de referencia (evita instabilidade numerica com dados shifted).

    Retorna psi_score, psi_status e psi_drift (bool).
    """
    # Recorta amostras ao intervalo da referencia (padrao de producao).
    # Evita instabilidade numerica com bins de borda de peso epsilon vs peso real.
    clipped = np.clip(new_samples, bins[0], bins[-1])
    new_counts, _ = np.histogram(clipped, bins=bins)

    # Proporcoes do novo lote com epsilon para evitar log(0)
    new_arr = (new_counts / new_counts.sum()) + 1e-9

    # Proporcoes de referencia (epsilon ja adicionado em prepare_reference.py)
    ref_arr = np.array(ref_proportions)

    min_len = min(len(ref_arr), len(new_arr))
    ref_arr = ref_arr[:min_len]
    new_arr = new_arr[:min_len]

    psi = float(np.sum((new_arr - ref_arr) * np.log(new_arr / ref_arr)))

    if psi >= PSI_ALERT:
        status = "ALERT"
    elif psi >= PSI_WARN:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "psi_score":  round(psi, 4),
        "psi_status": status,
        "psi_drift":  bool(psi >= PSI_WARN),
    }


def compute_jsd(ref_proportions: list, bins: list, new_samples: list) -> dict:
    """
    Jensen-Shannon Divergence: distancia simetrica entre distribuicoes.

    scipy.spatial.distance.jensenshannon retorna a RAIZ QUADRADA do JSD,
    que e a metrica de distancia JS (varia de 0 a 1).

    Usa bins estendidos com -inf/+inf para capturar valores fora do range
    de referencia (mesma logica do compute_psi).

    Retorna jsd_score, jsd_status e jsd_drift (bool).
    """
    # Recorta amostras ao intervalo da referencia (mesma logica do compute_psi).
    clipped = np.clip(new_samples, bins[0], bins[-1])
    new_counts, _ = np.histogram(clipped, bins=bins)

    # Proporcoes com epsilon para evitar divisao por zero
    new_arr = (new_counts / new_counts.sum()) + 1e-9

    # Proporcoes de referencia
    ref_arr = np.array(ref_proportions)

    min_len = min(len(ref_arr), len(new_arr))
    ref_arr = ref_arr[:min_len]
    new_arr = new_arr[:min_len]

    # Normaliza para que ambos somem 1 (requisito do JSD como distribuicao de probabilidade)
    ref_norm = ref_arr / ref_arr.sum()
    new_norm = new_arr / new_arr.sum()

    # jensenshannon retorna a distancia JS (raiz do JSD), nao o JSD em si
    jsd_dist = float(jensenshannon(ref_norm, new_norm))

    if jsd_dist >= JSD_ALERT:
        status = "ALERT"
    elif jsd_dist >= JSD_WARN:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "jsd_score":  round(jsd_dist, 4),
        "jsd_status": status,
        "jsd_drift":  bool(jsd_dist >= JSD_WARN),
    }


def detect_drift(ref_stats: dict, new_batch: list, features: list) -> dict:
    """
    Executa KS + PSI + JSD para cada feature e determina a severidade.

    Regra de severidade por feature:
        ALERT   — 2 ou mais metodos com drift
        WARNING — 1 metodo com drift
        OK      — nenhum metodo com drift
    """

    results = {}

    for feature in features:
        ref_samples     = ref_stats[feature]["samples"]
        ref_proportions = ref_stats[feature]["ref_proportions"]
        bins            = ref_stats[feature]["bins"]
        new_samples     = [row[feature] for row in new_batch]

        # Executa os tres detectores
        ks  = run_ks_test(ref_samples, new_samples)
        psi = compute_psi(ref_proportions, bins, new_samples)
        jsd = compute_jsd(ref_proportions, bins, new_samples)

        # Conta quantos metodos detectaram drift
        drift_votes = sum([ks["ks_drift"], psi["psi_drift"], jsd["jsd_drift"]])

        if drift_votes >= 2:
            severity = "ALERT"
        elif drift_votes == 1:
            severity = "WARNING"
        else:
            severity = "OK"

        new_arr = np.array(new_samples)
        results[feature] = {
            **ks, **psi, **jsd,
            "severity":       severity,
            "drift_detected": bool(drift_votes >= 1),
            "drift_votes":    drift_votes,       # quantos dos 3 metodos concordam
            "reference_mean": round(ref_stats[feature]["mean"], 3),
            "new_mean":       round(float(np.mean(new_arr)), 3),
        }

    drifted  = [f for f, r in results.items() if r["drift_detected"]]
    alerted  = [f for f, r in results.items() if r["severity"] == "ALERT"]
    warned   = [f for f, r in results.items() if r["severity"] == "WARNING"]

    # Status geral: mais grave entre todas as features
    if alerted:
        overall_status = "ALERT"
    elif warned:
        overall_status = "WARNING"
    else:
        overall_status = "OK"

    return {
        "overall_status":      overall_status,
        "drift_detected":      len(drifted) > 0,
        "features_alert":      alerted,
        "features_warning":    warned,
        "features_ok":         [f for f in features if f not in drifted],
        "features":            results,
    }
