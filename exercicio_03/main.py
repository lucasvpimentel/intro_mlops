"""
main.py — Exercicio 03: Wine Classifier
========================================
Ponto de entrada unico do projeto. Todos os comandos passam por aqui.

Este arquivo nao contem logica de ML — ele organiza os comandos e
delega para os modulos corretos em src/.

Comandos disponíveis:
    python main.py download              Baixa e prepara os dados
    python main.py features              Normaliza features e salva o scaler
    python main.py train                 Treina e salva o modelo
    python main.py evaluate              Avalia o modelo e gera evaluation.txt
    python main.py predict               Le input.json e gera output.csv
    python main.py pipeline              Executa tudo em sequencia
"""

import argparse  # biblioteca padrao para criar interfaces de linha de comando


def cmd_download(_args):
    """
    Comando: python main.py download
    Baixa raw.csv e ja normaliza os dados (chama build_features internamente).
    """
    from src.data.download_data import download
    download()


def cmd_features(_args):
    """
    Comando: python main.py features
    Normaliza raw.csv e salva scaler.joblib e processed.csv.
    Util para re-normalizar sem re-baixar os dados brutos.
    """
    from src.features.build_features import build
    build()


def cmd_train(_args):
    """
    Comando: python main.py train
    Treina o Random Forest e salva wine_model.joblib.
    """
    from src.models.train import train
    train()


def cmd_evaluate(_args):
    """
    Comando: python main.py evaluate
    Avalia o modelo no conjunto de teste e salva evaluation.txt.
    """
    from src.models.evaluate import evaluate
    evaluate()


def cmd_predict(_args):
    """
    Comando: python main.py predict
    Le data/input.json, classifica cada vinho e salva data/output.csv.
    """
    from src.models.predict import predict_batch
    predict_batch()


def cmd_pipeline(_args):
    """
    Comando: python main.py pipeline
    Executa o pipeline completo em 4 etapas, exibindo o progresso de cada uma.

    Etapas:
        1. download  -> raw.csv + processed.csv + scaler.joblib
        2. features  -> re-normaliza (redundante, mas deixa o pipeline explicito)
        3. train     -> wine_model.joblib
        4. evaluate  -> evaluation.txt
    """
    from src.data.download_data import download
    from src.features.build_features import build
    from src.models.train import train
    from src.models.evaluate import evaluate

    print("==> [1/4] Baixando dataset...")
    download()

    print("\n==> [2/4] Normalizando features...")
    build()

    print("\n==> [3/4] Treinando modelo...")
    train()

    print("\n==> [4/4] Avaliando modelo...")
    evaluate()

    print("\nPipeline concluido. Use 'python main.py predict' para inferencia em lote.")


def main():
    """
    Configura o parser de linha de comando e despacha para o comando correto.

    argparse cria automaticamente a saida do --help baseada nos
    add_parser() e add_argument() registrados abaixo.
    """

    # Parser principal com descricao e instrucoes completas no --help
    parser = argparse.ArgumentParser(
        description="Classificador de Qualidade de Vinhos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # exibe a docstring deste arquivo ao final do --help
    )

    # Grupo de subcomandos — o usuario deve escolher um
    sub = parser.add_subparsers(dest="comando", required=True)

    # Registra cada subcomando com descricao para o --help
    sub.add_parser("download", help="Baixa o dataset")
    sub.add_parser("features", help="Normaliza features e salva o scaler")
    sub.add_parser("train",    help="Treina e salva o modelo")
    sub.add_parser("evaluate", help="Avalia o modelo e gera evaluation.txt")
    sub.add_parser("predict",  help="Inferencia em lote: input.json -> output.csv")
    sub.add_parser("pipeline", help="download + features + train + evaluate")

    # Faz o parse dos argumentos fornecidos no terminal
    args = parser.parse_args()

    # Mapeia o nome do subcomando para a funcao correspondente e a executa
    {
        "download": cmd_download,
        "features": cmd_features,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "predict":  cmd_predict,
        "pipeline": cmd_pipeline,
    }[args.comando](args)  # ex: se args.comando == "train", chama cmd_train(args)


if __name__ == "__main__":
    main()
