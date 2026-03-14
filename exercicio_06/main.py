"""
main.py — Exercicio 06: Drift Monitor para Wine (Ex03)
=======================================================
Monitor mais completo: usa tres metodos simultaneos (KS + PSI + JSD)
e classifica cada feature por severidade de drift (OK / WARNING / ALERT).

Comandos:
    python main.py prepare
    python main.py simulate [--drift none|low|high] [--n 60]
    python main.py detect
    python main.py pipeline [--drift none|low|high]
"""

import argparse


def cmd_prepare(_args):
    from src.data.prepare_reference import prepare_reference
    prepare_reference()


def cmd_simulate(args):
    from src.simulation.generate_batch import generate_batch
    generate_batch(drift_level=args.drift, n_samples=args.n)


def cmd_detect(_args):
    import json
    import os
    from src.monitoring.detector import detect_drift
    from src.monitoring.report import save_report

    root       = os.path.dirname(__file__)
    ref_path   = os.path.join(root, "data", "reference_stats.json")
    batch_path = os.path.join(root, "data", "new_batch.json")

    for path in [ref_path, batch_path]:
        if not os.path.exists(path):
            print(f"Arquivo nao encontrado: {path}")
            return

    with open(ref_path,   "r", encoding="utf-8") as f: ref_stats = json.load(f)
    with open(batch_path, "r", encoding="utf-8") as f: new_batch = json.load(f)

    features = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280_od315", "proline",
    ]
    result = detect_drift(ref_stats, new_batch, features)
    save_report(result, batch_size=len(new_batch))


def cmd_pipeline(args):
    from src.data.prepare_reference import prepare_reference
    from src.simulation.generate_batch import generate_batch

    print("==> [1/3] Calculando estatisticas de referencia...")
    prepare_reference()
    print(f"\n==> [2/3] Gerando lote simulado (drift='{args.drift}')...")
    generate_batch(drift_level=args.drift, n_samples=args.n)
    print("\n==> [3/3] Detectando drift (KS + PSI + JSD)...")
    cmd_detect(args)


def main():
    parser = argparse.ArgumentParser(
        description="Drift Monitor — Wine Classifier (Ex03)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("prepare", help="Calcula estatisticas de referencia do Ex03")

    p_sim = sub.add_parser("simulate", help="Gera novo lote simulado")
    p_sim.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_sim.add_argument("--n", type=int, default=120)

    sub.add_parser("detect", help="Detecta drift com KS + PSI + JSD")

    p_pipe = sub.add_parser("pipeline", help="prepare + simulate + detect")
    p_pipe.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_pipe.add_argument("--n", type=int, default=120)

    args = parser.parse_args()
    {"prepare": cmd_prepare, "simulate": cmd_simulate,
     "detect": cmd_detect, "pipeline": cmd_pipeline}[args.comando](args)


if __name__ == "__main__":
    main()
