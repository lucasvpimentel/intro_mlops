"""
src/monitoring/report.py — Exercicio 04: Drift Monitor (Iris)
==============================================================
Responsabilidade: receber os resultados do detector, imprimir no
terminal de forma legivel e salvar o relatorio em JSON.
"""

import json
from datetime import datetime  # para registrar o timestamp do relatorio
from pathlib import Path


# Caminho padrao do relatorio de saida
ROOT        = Path(__file__).parent.parent
REPORT_PATH = ROOT / "data" / "drift_report.json"


def save_report(detection_result: dict, batch_size: int):
    """
    Imprime o relatorio de drift no terminal e salva em JSON.

    Parametros:
        detection_result (dict): saida da funcao detect_drift()
        batch_size       (int):  numero de amostras no novo lote
    """

    # Adiciona metadados ao relatorio: quando foi gerado e quantas amostras
    report = {
        "timestamp":   datetime.now().isoformat(),  # ex: "2026-03-13T14:30:00"
        "exercise":    "Iris Classifier (Ex01)",
        "batch_size":  batch_size,
        "method":      "Kolmogorov-Smirnov Test (p < 0.05)",
        **detection_result,  # inclui todos os campos do detector
    }

    # --- Exibe no terminal ---
    status_symbol = "ALERTA" if report["drift_detected"] else "OK"
    print("\n" + "=" * 55)
    print(f"  RELATORIO DE DRIFT — {report['exercise']}")
    print("=" * 55)
    print(f"  Status geral : [{status_symbol}] {report['overall_status']}")
    print(f"  Amostras     : {batch_size}")
    print(f"  Metodo       : {report['method']}")
    print("-" * 55)

    # Exibe os resultados por feature
    print(f"  {'Feature':<22} {'KS-stat':>8} {'p-value':>8} {'Status':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for feature, res in detection_result["features"].items():
        status = "DRIFT" if res["drift_detected"] else "ok"
        print(
            f"  {feature:<22} {res['ks_statistic']:>8.4f} "
            f"{res['ks_pvalue']:>8.4f} {status:>8}"
        )

    print("-" * 55)

    # Resumo das features com drift
    if detection_result["features_with_drift"]:
        print(f"  Features com drift: {', '.join(detection_result['features_with_drift'])}")
    else:
        print("  Nenhuma feature com drift detectado.")

    print("=" * 55)

    # --- Salva em JSON ---
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatorio salvo em: {REPORT_PATH}")
