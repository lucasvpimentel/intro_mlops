"""
src/monitoring/report.py — Exercicio 05: Drift Monitor (Diabetes)
==================================================================
Responsabilidade: exibir o relatorio de drift com resultados de KS e PSI
e salvar em JSON.
"""

import os
import json
from datetime import datetime

ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_PATH = os.path.join(ROOT, "data", "drift_report.json")

# Simbolos ASCII para cada status de PSI (compativel com cp1252/Windows)
PSI_SYMBOLS = {"OK": "+", "WARNING": "~", "ALERT": "!"}


def save_report(detection_result: dict, batch_size: int):
    """
    Imprime o relatorio de drift com KS e PSI no terminal e salva JSON.

    Parametros:
        detection_result (dict): saida de detect_drift()
        batch_size       (int):  numero de amostras no lote
    """

    report = {
        "timestamp":  datetime.now().isoformat(),
        "exercise":   "Diabetes Regressor (Ex02)",
        "batch_size": batch_size,
        "methods":    ["Kolmogorov-Smirnov (p < 0.05)", "PSI (< 0.1 OK | 0.1-0.2 WARNING | >= 0.2 ALERT)"],
        **detection_result,
    }

    status_label = {"OK": "OK", "WARNING": "ATENCAO", "ALERT": "ALERTA"}[report["overall_status"]]

    print("\n" + "=" * 70)
    print(f"  RELATORIO DE DRIFT — {report['exercise']}")
    print("=" * 70)
    print(f"  Status geral : [{status_label}] {report['overall_status']}")
    print(f"  Amostras     : {batch_size}")
    print(f"  PSI maximo   : {report['max_psi']:.4f}")
    print("-" * 70)
    print(f"  {'Feature':<8} {'KS-stat':>8} {'p-val':>8} {'KS':>6} | {'PSI':>8} {'PSI status':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*6}   {'-'*8} {'-'*12}")

    for feature, res in detection_result["features"].items():
        ks_flag  = "DRIFT" if res["ks_drift"]  else "ok"
        psi_sym  = PSI_SYMBOLS.get(res["psi_status"], "?")
        print(
            f"  {feature:<8} {res['ks_statistic']:>8.4f} {res['ks_pvalue']:>8.4f} {ks_flag:>6} | "
            f"{res['psi_score']:>8.4f} {psi_sym} {res['psi_status']:>10}"
        )

    print("-" * 70)
    if detection_result["features_with_drift"]:
        print(f"  Features com drift: {', '.join(detection_result['features_with_drift'])}")
    else:
        print("  Nenhuma feature com drift detectado.")
    print("=" * 70)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatorio salvo em: {REPORT_PATH}")
