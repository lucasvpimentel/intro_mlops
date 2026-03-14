"""
main.py — Exercicio 04: Drift Monitor para Iris (Ex01)
=======================================================
Monitora drift nos dados de entrada do Classificador de Iris.

Como funciona:
    1. prepare  -> le os dados de treino do Ex01 e salva estatisticas de referencia
    2. simulate -> gera um lote simulado com drift controlado
    3. detect   -> compara o lote contra a referencia (Teste KS)
    4. pipeline -> executa os tres passos em sequencia

Comandos:
    python main.py prepare
    python main.py simulate [--drift none|low|high] [--n 50]
    python main.py detect
    python main.py pipeline [--drift none|low|high]
"""

import argparse


def cmd_prepare(_args):
    """Calcula estatisticas de referencia a partir dos dados do Ex01."""
    from src.data.prepare_reference import prepare_reference
    prepare_reference()


def cmd_simulate(args):
    """Gera um novo lote simulado com o nivel de drift especificado."""
    from src.simulation.generate_batch import generate_batch
    generate_batch(drift_level=args.drift, n_samples=args.n)


def cmd_detect(_args):
    """
    Carrega a referencia e o lote atual, roda o detector KS
    e salva o relatorio.
    """
    import json
    from pathlib import Path
    from src.monitoring.detector import detect_drift
    from src.monitoring.report import save_report

    # Caminhos dos arquivos de entrada do detector
    root       = Path(__file__).parent
    ref_path   = root / "data" / "reference_stats.json"
    batch_path = root / "data" / "new_batch.json"

    # Verifica se os dois arquivos existem
    for path in [ref_path, batch_path]:
        if not path.exists():
            print(f"Arquivo nao encontrado: {path}")
            print("Execute: python main.py pipeline")
            return

    # Carrega referencia e lote
    with open(ref_path,   "r", encoding="utf-8") as f:
        ref_stats = json.load(f)
    with open(batch_path, "r", encoding="utf-8") as f:
        new_batch = json.load(f)

    # Features monitoradas
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Executa a deteccao e gera o relatorio
    result = detect_drift(ref_stats, new_batch, features)
    save_report(result, batch_size=len(new_batch))


def cmd_pipeline(args):
    """Executa prepare + simulate + detect em sequencia."""
    from src.data.prepare_reference import prepare_reference
    from src.simulation.generate_batch import generate_batch

    print("==> [1/3] Calculando estatisticas de referencia...")
    prepare_reference()

    print(f"\n==> [2/3] Gerando lote simulado (drift='{args.drift}')...")
    generate_batch(drift_level=args.drift, n_samples=args.n)

    print("\n==> [3/3] Detectando drift...")
    cmd_detect(args)


def main():
    """Configura o CLI e despacha para o comando correto."""
    parser = argparse.ArgumentParser(
        description="Drift Monitor — Iris Classifier (Ex01)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    # prepare: sem argumentos extras
    sub.add_parser("prepare", help="Calcula estatisticas de referencia do Ex01")

    # simulate: aceita --drift e --n
    p_sim = sub.add_parser("simulate", help="Gera novo lote simulado")
    p_sim.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_sim.add_argument("--n", type=int, default=50, help="Numero de amostras")

    # detect: sem argumentos extras
    sub.add_parser("detect", help="Detecta drift no lote atual")

    # pipeline: aceita --drift e --n
    p_pipe = sub.add_parser("pipeline", help="prepare + simulate + detect")
    p_pipe.add_argument("--drift", choices=["none", "low", "high"], default="none")
    p_pipe.add_argument("--n", type=int, default=50, help="Numero de amostras")

    args = parser.parse_args()
    {
        "prepare":  cmd_prepare,
        "simulate": cmd_simulate,
        "detect":   cmd_detect,
        "pipeline": cmd_pipeline,
    }[args.comando](args)


if __name__ == "__main__":
    main()
