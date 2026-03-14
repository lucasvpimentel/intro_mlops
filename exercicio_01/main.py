"""
main.py — Exercicio 01: Iris Classifier
========================================
Ponto de entrada unico do projeto. Todos os comandos passam por aqui.

Este arquivo nao contem logica de ML — ele apenas organiza os comandos
e delega para os modulos corretos em src/.

Comandos disponíveis:
    python main.py download              Baixa o dataset e salva raw.csv
    python main.py train                 Treina o modelo
    python main.py predict 5.1 3.5 1.4 0.2   Preve a especie de uma flor
    python main.py pipeline              Executa download + treino em sequencia
"""

import sys       # acesso a sys.exit para encerrar com erro
import argparse  # biblioteca para criar interfaces de linha de comando


def cmd_download(_args):
    """
    Comando: python main.py download
    Delega para src/data/download_data.py.
    """
    from src.data.download_data import download  # importa a funcao de download
    download()                                    # executa o download


def cmd_train(_args):
    """
    Comando: python main.py train
    Delega para src/models/train.py.
    """
    from src.models.train import train  # importa a funcao de treino
    train()                              # executa o treino


def cmd_predict(args):
    """
    Comando: python main.py predict <sepal_length> <sepal_width> <petal_length> <petal_width>
    Recebe 4 medidas numericas e imprime a especie prevista com a confianca.

    Parametros do argparse:
        args.medidas: lista com os 4 valores fornecidos pelo usuario
    """
    from src.models.predict import predict  # importa a funcao de predicao

    # Garante que exatamente 4 valores foram fornecidos
    if len(args.medidas) != 4:
        print("Erro: forneca exatamente 4 medidas: sepal_length sepal_width petal_length petal_width")
        sys.exit(1)

    # Tenta converter cada valor para float; exibe erro se nao for numerico
    try:
        valores = [float(v) for v in args.medidas]  # converte strings para numeros
    except ValueError:
        print("Erro: todos os valores devem ser numeros decimais.")
        sys.exit(1)

    # Desempacota a lista de 4 valores como 4 argumentos separados para predict()
    species, confidence = predict(*valores)

    # Exibe o resultado formatado com o mesmo estilo do restante do projeto
    print("\n" + "=" * 40)
    print("  PREDICAO — Iris Classifier")
    print("=" * 40)
    print(f"  Especie    : {species}")
    print(f"  Confianca  : {confidence:.1f}%")
    print("=" * 40)


def cmd_pipeline(_args):
    """
    Comando: python main.py pipeline
    Executa download e treino em sequencia, exibindo o progresso.
    """
    from src.data.download_data import download  # passo 1: dados
    from src.models.train import train            # passo 2: treino

    print("==> [1/2] Baixando dataset...")
    download()

    print("\n==> [2/2] Treinando modelo...")
    train()

    print("\nPipeline concluido. Use 'python main.py predict <4 medidas>' para inferencia.")


def main():
    """
    Configura o parser de linha de comando e despacha para o comando correto.

    argparse e a biblioteca padrao do Python para criar CLIs.
    Cada subcomando (download, train, predict, pipeline) e registrado
    como um subparser e mapeado para uma funcao cmd_*.
    """

    # Cria o parser principal com descricao exibida no --help
    parser = argparse.ArgumentParser(
        description="Classificador de Especies Iris",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # exibe a docstring deste arquivo no --help
    )

    # Cria o grupo de subcomandos; required=True exige que um seja fornecido
    sub = parser.add_subparsers(dest="comando", required=True)

    # Registra cada subcomando com sua descricao para o --help
    sub.add_parser("download", help="Baixa o dataset e salva raw.csv")
    sub.add_parser("train",    help="Treina e salva o modelo")
    sub.add_parser("pipeline", help="Executa download + treino em sequencia")

    # O subcomando "predict" precisa de argumentos adicionais (as 4 medidas)
    p_predict = sub.add_parser("predict", help="Preve a especie de uma flor")
    p_predict.add_argument(
        "medidas",  # nome interno do argumento
        nargs=4,    # espera exatamente 4 valores
        metavar=("sepal_length", "sepal_width", "petal_length", "petal_width"),
        help="4 medidas em cm",
    )

    # Faz o parse dos argumentos fornecidos pelo usuario no terminal
    args = parser.parse_args()

    # Dicionario que mapeia o nome do subcomando para a funcao correspondente
    # args.comando contem o subcomando escolhido (ex: "train")
    {
        "download": cmd_download,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "pipeline": cmd_pipeline,
    }[args.comando](args)  # chama a funcao passando os argumentos


# Ponto de entrada quando executado diretamente
if __name__ == "__main__":
    main()
