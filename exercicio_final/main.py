"""
main.py — Exercicio Final: Penguins MLOps
==========================================
Orquestrador central do projeto. Todos os comandos passam por aqui.

Uso:
    python main.py download                  # Baixa o dataset
    python main.py split                     # Divide em treino/teste
    python main.py train                     # Treina classificador + regressor
    python main.py evaluate                  # Avalia no conjunto de teste
    python main.py predict --bill-length 39.1 --bill-depth 18.7
                           --flipper-length 181.0 --sex male --island Torgersen
    python main.py predict-batch             # Predicao em lote (data/samples/new_penguins.json)
    python main.py pipeline                  # download + split + train + evaluate
"""

import argparse
import sys


def cmd_download(_args):
    """Baixa o dataset Palmer Penguins e salva em data/raw/penguins.csv."""
    from src.data_loader import download_data
    download_data()


def cmd_split(_args):
    """Divide o dataset em treino (80%) e teste (20%) com estratificacao por especie."""
    from src.data_loader import split_data
    split_data()


def cmd_train(_args):
    """Treina o classificador de especie e o regressor de peso. Salva modelos em models/."""
    from src.trainer import train
    train()


def cmd_evaluate(_args):
    """Avalia os modelos no conjunto de teste. Salva graficos em reports/."""
    from src.evaluator import evaluate
    evaluate()


def cmd_predict(args):
    """
    Prediz especie e peso de um unico pinguim com as medicoes fornecidas.

    Exemplo:
        python main.py predict --bill-length 39.1 --bill-depth 18.7
                               --flipper-length 181.0 --sex male --island Torgersen
    """
    from src.inference import predict_single

    resultado = predict_single(
        bill_length_mm    = args.bill_length,
        bill_depth_mm     = args.bill_depth,
        flipper_length_mm = args.flipper_length,
        sex               = args.sex,
        island            = args.island,
    )

    print("\n" + "=" * 40)
    print("  PREDICAO")
    print("=" * 40)
    print(f"  Especie        : {resultado['especie']}")
    print(f"  Confianca      : {resultado['confianca_pct']}%")
    print(f"  Peso estimado  : {resultado['peso_estimado_g']} g")
    print("=" * 40)


def cmd_predict_batch(args):
    """
    Predicao em lote a partir do arquivo JSON em data/samples/new_penguins.json.
    """
    from src.inference import predict_batch, SAMPLES_PATH

    path = args.input if hasattr(args, "input") and args.input else SAMPLES_PATH
    results = predict_batch(path)

    print(f"\nTotal: {len(results)} predicoes concluidas.")


def cmd_pipeline(_args):
    """Executa o pipeline completo: download → split → train → evaluate."""
    from src.data_loader import download_data, split_data
    from src.trainer import train
    from src.evaluator import evaluate

    print("=" * 60)
    print("  PIPELINE COMPLETO — Penguins MLOps")
    print("=" * 60)

    print("\n[1/4] Baixando dataset...")
    download_data()

    print("\n[2/4] Dividindo em treino e teste...")
    split_data()

    print("\n[3/4] Treinando modelos...")
    train()

    print("\n[4/4] Avaliando desempenho...")
    evaluate()

    print("\nPipeline concluido com sucesso!")


def main():
    """Ponto de entrada principal. Define todos os subcomandos do CLI."""

    parser = argparse.ArgumentParser(
        description="Penguins MLOps — Classificacao de Especie e Estimativa de Peso",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Cria o container de subcomandos
    sub = parser.add_subparsers(dest="comando", required=True)

    # Subcomando: download
    sub.add_parser("download", help="Baixa o dataset Palmer Penguins")

    # Subcomando: split
    sub.add_parser("split", help="Divide dataset em treino/teste")

    # Subcomando: train
    sub.add_parser("train", help="Treina classificador e regressor")

    # Subcomando: evaluate
    sub.add_parser("evaluate", help="Avalia modelos no conjunto de teste")

    # Subcomando: predict (predicao individual)
    p_pred = sub.add_parser("predict", help="Prediz especie e peso de um pinguim")
    p_pred.add_argument("--bill-length",    type=float, required=True,
                        help="Comprimento do bico em mm (ex: 39.1)")
    p_pred.add_argument("--bill-depth",     type=float, required=True,
                        help="Profundidade do bico em mm (ex: 18.7)")
    p_pred.add_argument("--flipper-length", type=float, required=True,
                        help="Comprimento da asa em mm (ex: 181.0)")
    p_pred.add_argument("--sex",    type=str, required=True, choices=["male", "female"],
                        help="Sexo do pinguim")
    p_pred.add_argument("--island", type=str, required=True,
                        choices=["Biscoe", "Dream", "Torgersen"],
                        help="Ilha onde foi coletado")

    # Subcomando: predict-batch (predicao em lote via JSON)
    p_batch = sub.add_parser("predict-batch", help="Predicao em lote via JSON")
    p_batch.add_argument("--input", type=str, default=None,
                         help="Caminho do JSON de entrada (padrao: data/samples/new_penguins.json)")

    # Subcomando: pipeline (tudo de uma vez)
    sub.add_parser("pipeline", help="Executa download + split + train + evaluate")

    # Parse dos argumentos e despacho para a funcao correta
    args = parser.parse_args()

    comandos = {
        "download":      cmd_download,
        "split":         cmd_split,
        "train":         cmd_train,
        "evaluate":      cmd_evaluate,
        "predict":       cmd_predict,
        "predict-batch": cmd_predict_batch,
        "pipeline":      cmd_pipeline,
    }

    comandos[args.comando](args)


if __name__ == "__main__":
    main()
