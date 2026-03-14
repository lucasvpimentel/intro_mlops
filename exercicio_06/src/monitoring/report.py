"""
src/monitoring/report.py — Exercicio 06: Drift Monitor (Wine)
=============================================================
Responsabilidade: exibir relatorio completo com resultados de KS, PSI
e JSD por feature, com codificacao visual de severidade.
"""

import os
import json
from datetime import datetime

ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_PATH = os.path.join(ROOT, "data", "drift_report.json")

# Icones de severidade para facilitar leitura visual
SEVERITY_ICON = {"OK": " OK ", "WARNING": "WARN", "ALERT": "ALRT"}


def save_report(detection_result: dict, batch_size: int):
    """
    Imprime relatorio tri-metodo (KS, PSI, JSD) e salva JSON.

    A tabela exibe, por feature:
        - KS: estatistica e p-value
        - PSI: score e status
        - JSD: score e status
        - Votos: quantos dos 3 metodos detectaram drift
        - Severidade final

    Parametros:
        detection_result (dict): saida de detect_drift()
        batch_size       (int):  numero de amostras no lote
    """

    report = {
        "timestamp":  datetime.now().isoformat(),
        "exercise":   "Wine Classifier (Ex03)",
        "batch_size": batch_size,
        "methods": {
            "KS":  "p < 0.05 = drift",
            "PSI": "< 0.1 OK | 0.1-0.2 WARNING | >= 0.2 ALERT",
            "JSD": "< 0.08 OK | 0.08-0.15 WARNING | >= 0.15 ALERT",
        },
        **detection_result,
    }

    overall = report["overall_status"]
    print("\n" + "=" * 80)
    print(f"  RELATORIO DE DRIFT — {report['exercise']}")
    print("=" * 80)
    print(f"  Status geral  : [{overall}]")
    print(f"  Amostras      : {batch_size}")
    if report["features_alert"]:
        print(f"  Em ALERTA     : {', '.join(report['features_alert'])}")
    if report["features_warning"]:
        print(f"  Em WARNING    : {', '.join(report['features_warning'])}")
    print("-" * 80)

    # Cabecalho da tabela
    print(
        f"  {'Feature':<24} "
        f"{'KS-stat':>7} {'p-val':>6} "
        f"{'PSI':>6} {'PSI-st':>7} "
        f"{'JSD':>6} {'JSD-st':>7} "
        f"{'Votos':>5} {'Sev':>6}"
    )
    print(f"  {'-'*24} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*5} {'-'*6}")

    for feature, res in detection_result["features"].items():
        sev_icon = SEVERITY_ICON[res["severity"]]
        print(
            f"  {feature:<24} "
            f"{res['ks_statistic']:>7.4f} {res['ks_pvalue']:>6.4f} "
            f"{res['psi_score']:>6.4f} {res['psi_status']:>7} "
            f"{res['jsd_score']:>6.4f} {res['jsd_status']:>7} "
            f"{res['drift_votes']:>5} {sev_icon:>6}"
        )

    print("=" * 80)
    print("  Legenda votos: numero de metodos (KS/PSI/JSD) que detectaram drift")
    print("  Severidade: OK = 0 votos | WARNING = 1 voto | ALERT = 2+ votos")
    print("=" * 80)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatorio salvo em: {REPORT_PATH}")
