"""
src/data/download_data.py — Exercicio 03: Wine Classifier
==========================================================
Responsabilidade: baixar o Wine Dataset e deixar os dados completamente
prontos para o treino.

Principio de Limpeza:
    Este script nao apenas baixa o CSV bruto — ele tambem chama
    build_features.py para normalizar os dados e salvar o scaler.
    Ao final, processed.csv e scaler.joblib estao disponiveis,
    e o treino pode comecar imediatamente.

Como executar diretamente:
    python src/data/download_data.py
"""

from sklearn.datasets import load_wine  # dataset classico de classificacao de vinhos
import pandas as pd  # manipulacao de dados tabulares
import sys           # adicionar raiz ao path de busca do Python
from pathlib import Path

# Nomes das 13 features quimicas do vinho — substituem os nomes originais do sklearn
FEATURES = [
    "alcohol",              # teor alcoolico
    "malic_acid",           # acido malico
    "ash",                  # cinzas
    "alcalinity_of_ash",    # alcalinidade das cinzas
    "magnesium",            # magnesio
    "total_phenols",        # fenois totais
    "flavanoids",           # flavanoides
    "nonflavanoid_phenols", # fenois nao-flavanoides
    "proanthocyanins",      # proantocianinas
    "color_intensity",      # intensidade de cor
    "hue",                  # matiz
    "od280_od315",          # razao OD280/OD315 (absorbancia UV)
    "proline",              # prolina (aminoacido)
]

# Caminho absoluto da raiz do projeto
ROOT = Path(__file__).parent.parent.parent

# Garante que imports "from src.features..." funcionem ao rodar diretamente
sys.path.insert(0, str(ROOT))


def download():
    """
    Carrega o Wine Dataset do scikit-learn e deixa tudo pronto para treino.

    O dataset tem 178 amostras de vinhos de 3 cultivares diferentes,
    com 13 features quimicas cada. As classes sao:
        class_0: 59 amostras
        class_1: 71 amostras
        class_2: 48 amostras

    Etapas executadas:
        1. Carrega e renomeia as colunas para nomes legíveis
        2. Adiciona a coluna 'cultivar' com o nome da classe
        3. Salva raw.csv
        4. Chama build_features para gerar processed.csv e scaler.joblib

    Nao recebe parametros e nao retorna nada.
    """

    # Carrega o dataset como DataFrame (as_frame=True)
    wine = load_wine(as_frame=True)

    # Copia o frame para nao alterar o objeto original do sklearn
    df = wine.frame.copy()

    # Substitui todos os nomes de coluna pelos nomes definidos em FEATURES + "target"
    # O sklearn usa nomes como "alcohol" ja neste dataset, mas renomeamos
    # explicitamente para garantir consistencia e remover acentos
    df.columns = FEATURES + ["target"]  # 13 features + coluna alvo numerica

    # Cria a coluna 'cultivar' com o nome legivel da classe
    # {0: 'class_0', 1: 'class_1', 2: 'class_2'} mapeia numero para string
    df["cultivar"] = df["target"].map({i: f"class_{i}" for i in range(3)})

    # Monta o caminho de saida para data/raw.csv
    out_path = ROOT / "data" / "raw.csv"

    # Salva sem o indice do pandas
    df.to_csv(out_path, index=False)
    print(f"Dataset salvo em: {out_path}")
    print(f"Shape: {df.shape}")

    # value_counts() conta amostras por cultivar; .to_dict() converte para dicionario
    print(f"Classes: {df['cultivar'].value_counts().to_dict()}")

    # Principio de Limpeza: deixa os dados prontos (normaliza e salva o scaler)
    from src.features.build_features import build
    build()


if __name__ == "__main__":
    download()
