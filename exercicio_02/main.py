"""
main.py — Exercicio 02: Diabetes Regressor
===========================================
Ponto de entrada unico do projeto. Todos os comandos passam por aqui.

Este arquivo organiza os comandos e delega para os modulos em src/.
Nao contem logica de ML.

Comandos disponíveis:
    python main.py download                            Baixa e prepara os dados
    python main.py features                            Normaliza e salva o scaler
    python main.py train [--model ridge|rf]            Treina o modelo
    python main.py predict age sex bmi bp s1...        Estima progressao
    python main.py pipeline [--model ridge|rf]         Executa tudo em sequencia
"""

import sys       # sys.exit para erros
import argparse  # interface de linha de comando

# Lista de features usada para validacao e exibicao no --help
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def cmd_download(_args):
    """
    Comando: python main.py download
    Baixa raw.csv e ja normaliza os dados (processed.csv + scaler.joblib).
    """
    from src.data.download_data import download
    download()


def cmd_features(_args):
    """
    Comando: python main.py features
    Normaliza raw.csv e salva scaler.joblib e processed.csv.
    Util para re-normalizar sem re-baixar os dados.
    """
    from src.features.build_features import build
    build()


def cmd_train(args):
    """
    Comando: python main.py train [--model ridge|rf]
    Treina o modelo escolhido e salva em data/models/.

    Parametros do argparse:
        args.model (str): 'ridge' ou 'rf'
    """
    from src.models.train import train
    train(args.model)  # passa o tipo de modelo escolhido pelo usuario


def cmd_predict(args):
    """
    Comando: python main.py predict <10 valores>
    Estima a progressao da diabetes para um paciente.

    Parametros do argparse:
        args.valores (list): lista de 10 strings com os valores das features
    """
    from src.models.predict import predict

    # Garante que exatamente 10 valores foram fornecidos
    if len(args.valores) != 10:
        print(f"Erro: forneca exatamente 10 valores ({', '.join(FEATURES)})")
        sys.exit(1)

    # Converte strings para float, tratando erros de formato
    try:
        valores = [float(v) for v in args.valores]
    except ValueError:
        print("Erro: todos os valores devem ser numeros decimais.")
        sys.exit(1)

    # Chama a funcao de predicao e exibe o resultado formatado
    result = predict(valores)
    print(f"\nProgressao estimada da diabetes (1 ano): {result:.1f}")
    print("(Escala: ~25 = baixa progressao | ~346 = alta progressao)")


def cmd_pipeline(args):
    """
    Comando: python main.py pipeline [--model ridge|rf]
    Executa download + features + treino em sequencia.

    Nota: download() ja chama build() internamente, mas chamamos
    build() explicitamente aqui para deixar o pipeline legivel.

    Parametros do argparse:
        args.model (str): 'ridge' ou 'rf'
    """
    from src.data.download_data import download
    from src.features.build_features import build
    from src.models.train import train

    print("==> [1/3] Baixando dataset...")
    download()

    print("\n==> [2/3] Normalizando features e salvando scaler...")
    build()

    print(f"\n==> [3/3] Treinando modelo ({args.model})...")
    train(args.model)

    print("\nPipeline concluido. Use 'python main.py predict <10 valores>' para inferencia.")


def main():
    """
    Configura o parser de linha de comando e despacha para o comando correto.
    """

    # Parser principal
    parser = argparse.ArgumentParser(
        description="Estimador de Progressao de Diabetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Grupo de subcomandos
    sub = parser.add_subparsers(dest="comando", required=True)

    # Subcomandos simples (sem argumentos extras)
    sub.add_parser("download", help="Baixa o dataset e prepara os dados")
    sub.add_parser("features", help="Normaliza features e salva o scaler")

    # Subcomando train: aceita --model para escolher o algoritmo
    p_train = sub.add_parser("train", help="Treina e salva o modelo")
    p_train.add_argument(
        "--model", choices=["ridge", "rf"], default="ridge",
        help="Algoritmo: ridge (padrao) ou rf (Random Forest)"
    )

    # Subcomando predict: requer os 10 valores das features
    p_predict = sub.add_parser("predict", help="Estima progressao para um paciente")
    p_predict.add_argument(
        "valores",  # nome interno da lista de argumentos
        nargs=10,   # espera exatamente 10 valores posicionais
        metavar=tuple(FEATURES),  # exibe os nomes no --help
        help="10 features clinicas",
    )

    # Subcomando pipeline: tambem aceita --model
    p_pipeline = sub.add_parser("pipeline", help="Executa download + features + treino")
    p_pipeline.add_argument(
        "--model", choices=["ridge", "rf"], default="ridge",
        help="Algoritmo: ridge (padrao) ou rf (Random Forest)"
    )

    # Faz o parse e chama a funcao correspondente ao subcomando
    args = parser.parse_args()
    {
        "download": cmd_download,
        "features": cmd_features,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "pipeline": cmd_pipeline,
    }[args.comando](args)


if __name__ == "__main__":
    main()
