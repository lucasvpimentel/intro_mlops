"""
src/models/predict.py — Exercicio 01: Iris Classifier
======================================================
Responsabilidade: carregar o modelo treinado e fazer predicoes.

Principio de Isolamento:
    Este modulo NAO sabe como o modelo foi treinado.
    Ele apenas carrega o arquivo .joblib gerado pelo train.py
    e usa a interface padrao do scikit-learn (.predict, .predict_proba).
    Qualquer mudanca no algoritmo de treino nao afeta este arquivo.
"""

import os     # manipulacao de caminhos
import sys    # encerrar o programa com mensagem de erro
import joblib # carregar o modelo salvo em disco
import pandas as pd  # criar DataFrame para passar ao modelo

# Caminho raiz do projeto (dois niveis acima deste arquivo)
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Caminho para o arquivo do modelo salvo pelo train.py
MODEL_PATH = os.path.join(ROOT, "data", "models", "iris_model.joblib")

# Nomes das colunas que o modelo espera receber — mesma ordem do treino
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_model():
    """
    Carrega o modelo serializado do disco.

    Verifica se o arquivo existe antes de tentar carregar.
    Se nao existir, exibe mensagem orientando o usuario e encerra.

    Retorna:
        model: objeto do scikit-learn com os metodos .predict() e .predict_proba()
    """

    # Verifica se o arquivo do modelo foi criado pelo train.py
    if not os.path.exists(MODEL_PATH):
        print("Modelo nao encontrado. Execute: python main.py train")
        sys.exit(1)  # encerra o programa com codigo de erro 1

    # Desserializa e retorna o modelo carregado da memoria
    return joblib.load(MODEL_PATH)


def predict(sepal_length: float, sepal_width: float,
            petal_length: float, petal_width: float):
    """
    Recebe as 4 medidas de uma flor e retorna a especie prevista.

    Parametros:
        sepal_length (float): comprimento da sepala em cm
        sepal_width  (float): largura da sepala em cm
        petal_length (float): comprimento da petala em cm
        petal_width  (float): largura da petala em cm

    Retorna:
        species    (str):   nome da especie prevista (ex: 'setosa')
        confidence (float): percentual de certeza do modelo (0 a 100)
    """

    # Carrega o modelo do disco
    model = load_model()

    # Cria um DataFrame de uma linha com os valores recebidos
    # Usar DataFrame (ao inves de lista simples) evita warnings do sklearn
    # pois o modelo foi treinado com dados nomeados
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,  # associa cada valor ao nome correto da coluna
    )

    # .predict() retorna a classe mais provavel — pegamos o primeiro elemento [0]
    species = model.predict(features)[0]

    # .predict_proba() retorna a probabilidade para cada classe
    # .max() pega a maior probabilidade e multiplicamos por 100 para ter percentual
    confidence = model.predict_proba(features)[0].max() * 100

    return species, confidence


if __name__ == "__main__":
    # Exemplos de uso direto: python src/models/predict.py
    exemplos = [
        (5.1, 3.5, 1.4, 0.2, "setosa"),
        (6.0, 2.7, 5.1, 1.6, "versicolor"),
        (6.7, 3.3, 5.7, 2.5, "virginica"),
    ]

    print("\n" + "=" * 40)
    print("  PREDICAO — Iris Classifier")
    print("=" * 40)
    for sl, sw, pl, pw, esperado in exemplos:
        especie, confianca = predict(sl, sw, pl, pw)
        print(f"  Entrada : {sl} {sw} {pl} {pw}")
        print(f"  Especie : {especie}  (esperado: {esperado})")
        print(f"  Confianca: {confianca:.1f}%")
        print("-" * 40)
    print("=" * 40)
