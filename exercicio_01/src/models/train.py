"""
src/models/train.py — Exercicio 01: Iris Classifier
=====================================================
Responsabilidade: treinar um modelo de Regressao Logistica no dataset Iris
e salvar o modelo treinado em disco para uso na inferencia.

Principio de Reprodutibilidade:
    Se o arquivo de dados (raw.csv) nao existir, este script baixa
    os dados automaticamente antes de treinar. Basta rodar este arquivo
    e ele cuida de tudo, sem necessidade de etapas manuais anteriores.

Como executar diretamente:
    python src/models/train.py
"""

import sys     # manipulacao do caminho de busca de modulos Python
from pathlib import Path
import joblib  # salvar e carregar objetos Python em disco (mais eficiente que pickle para sklearn)
import pandas as pd  # leitura e manipulacao do CSV

# Modelos e utilitarios do scikit-learn
from sklearn.linear_model import LogisticRegression  # o algoritmo de classificacao
from sklearn.model_selection import train_test_split  # divide dados em treino e teste
from sklearn.metrics import classification_report     # relatorio de precisao/recall/f1

# ROOT e o caminho absoluto da raiz do projeto (dois niveis acima deste arquivo)
ROOT = Path(__file__).parent.parent

# Adiciona a raiz ao path de busca do Python para que imports como
# "from src.data..." funcionem mesmo ao rodar este arquivo diretamente
sys.path.insert(0, str(ROOT))

# Caminhos dos arquivos que este script le e escreve
DATA_PATH  = ROOT / "data" / "raw.csv"                    # entrada: dados brutos
MODEL_PATH = ROOT / "data" / "models" / "iris_model.joblib" # saida: modelo serializado

# Lista das colunas usadas como entrada (features) do modelo
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Nome da coluna que queremos prever (variavel alvo)
TARGET = "species"


def train():
    """
    Executa o pipeline completo de treinamento:
        1. Garante que os dados existem (baixa se necessario)
        2. Le o CSV e separa features do alvo
        3. Divide em conjuntos de treino (80%) e teste (20%)
        4. Treina uma Regressao Logistica
        5. Imprime o relatorio de classificacao
        6. Salva o modelo em data/models/iris_model.joblib

    Nao recebe parametros e nao retorna nada.
    """

    # Reprodutibilidade: se o arquivo de dados nao existe, baixa automaticamente
    if not DATA_PATH.exists():
        print("raw.csv nao encontrado. Baixando dataset automaticamente...")
        from src.data.download_data import download  # importa aqui para evitar importacao circular
        download()

    # Cria a pasta data/models/ se ela ainda nao existir
    # exist_ok=True evita erro caso a pasta ja exista
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Le o CSV e carrega como DataFrame do pandas
    df = pd.read_csv(DATA_PATH)

    # Separa as colunas de entrada (X) da coluna alvo (y)
    X = df[FEATURES]  # DataFrame com as 4 medidas
    y = df[TARGET]    # Serie com o nome da especie

    # Divide os dados: 80% para treinar o modelo, 20% para avaliar depois
    # random_state=42 garante que a divisao e sempre a mesma (reproducibilidade)
    # stratify=y garante proporcao igual de cada classe nos dois conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cria o modelo de Regressao Logistica
    # max_iter=200: numero maximo de iteracoes para convergencia do otimizador
    # random_state=42: garante resultados reproduziveis
    model = LogisticRegression(max_iter=200, random_state=42)

    # Treina o modelo: aprende os pesos que melhor separam as 3 classes
    model.fit(X_train, y_train)

    # Avalia o modelo no conjunto de TESTE (dados que o modelo nunca viu)
    # classification_report mostra precisao, recall e f1-score por classe
    print("=== Relatorio de Classificacao ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Serializa (salva) o modelo treinado em disco usando joblib
    # O arquivo .joblib pode ser carregado depois com joblib.load()
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


# Ponto de entrada quando executado diretamente via terminal
if __name__ == "__main__":
    train()
