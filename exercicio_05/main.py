"""
main.py — Exercicio 05: Drift Monitor para Diabetes (Ex02)
===========================================================
Monitora drift nos dados de entrada do Estimador de Progressao de Diabetes.
Usa Teste KS + PSI para deteccao.

Comandos:
    python main.py prepare
    python main.py simulate [--drift none|low|high] [--n 100]
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
    from pathlib import Path
    from src.monitoring.detector import detect_drift
    from src.monitoring.report import save_report

    root       = Path(__file__).parent
    ref_path   = root / "data" / "reference_stats.json"
    batch_path = root / "data" / "new_batch.json"

    for path in [ref_path, batch_path]:
        if not path.exists():
            print(f"Arquivo nao encontrado: {path}")
            return

    with open(ref_path,   "r", encoding="utf-8") as f: ref_stats = json.load(f)
    with open(batch_path, "r", encoding="utf-8") as f: new_batch = json.load(f)

    features = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    result   = detect_drift(ref_stats, new_batch, features)
    save_report(result, batch_size=len(new_batch))


def cmd_pipeline(args):
    from src.data.prepare_reference import prepare_reference
    from src.simulation.generate_batch import generate_batch

    print("==> [1/3] Calculando estatisticas de referencia...")
    prepare_reference()
    print(f"\n==> [2/3] Gerando lote simulado (drift='{args.drift}')...")
    generate_batch(drift_level=args.drift, n_samples=args.n)
    print("\n==> [3/3] Detectando drift (KS + PSI)...")
    cmd_detect(args)


def main():
    parser = argparse.ArgumentParser(
        description="Drift Monitor — Diabetes Regressor (Ex02)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("prepare", help="Calcula estatisticas de referencia do Ex02")

    p_sim = sub.add_parser("simulate", help="Gera novo lote simulado")
    p_sim.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_sim.add_argument("--n", type=int, default=100)

    sub.add_parser("detect", help="Detecta drift com KS + PSI")

    p_pipe = sub.add_parser("pipeline", help="prepare + simulate + detect")
    p_pipe.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_pipe.add_argument("--n", type=int, default=100)

    args = parser.parse_args()
    {"prepare": cmd_prepare, "simulate": cmd_simulate,
     "detect": cmd_detect, "pipeline": cmd_pipeline}[args.comando](args)


if __name__ == "__main__":
    main()
