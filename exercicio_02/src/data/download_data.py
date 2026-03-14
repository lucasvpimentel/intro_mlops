"""
src/data/download_data.py — Exercicio 02: Diabetes Regressor
=============================================================
Responsabilidade: baixar o dataset Diabetes e deixar os dados
completamente prontos para o treino.

Principio de Limpeza:
    Este script nao apenas baixa o CSV bruto — ele tambem chama
    build_features.py para normalizar os dados e salvar o scaler.
    Ao final, processed.csv e scaler.joblib estao disponiveis,
    e o treino pode comecar imediatamente sem etapas extras.

Como executar diretamente:
    python src/data/download_data.py
"""

# sklearn contem o dataset Diabetes classico de regressao
from sklearn.datasets import load_diabetes

import pandas as pd  # para salvar o DataFrame como CSV
import os            # para construir caminhos de arquivo
import sys           # para adicionar a raiz do projeto ao path de busca

# Caminho absoluto da raiz do projeto (dois niveis acima deste arquivo)
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Garante que imports como "from src.features..." funcionem ao rodar diretamente
sys.path.insert(0, ROOT)


def download():
    """
    Carrega o Diabetes Dataset do scikit-learn e deixa tudo pronto para treino.

    O dataset tem 442 pacientes com 10 features clinicas cada:
        age, sex, bmi, bp, s1 (colesterol), s2 (LDL), s3 (HDL),
        s4 (TCH), s5 (triglicerideos), s6 (glicose)
    Todas as features ja vem pre-normalizadas pelo sklearn (media 0).
    O alvo ('progression') e um numero continuo: quanto a doenca progrediu
    em 1 ano (escala de 25 a 346).

    Etapas executadas:
        1. Baixa e salva raw.csv
        2. Chama build_features para gerar processed.csv e scaler.joblib

    Nao recebe parametros e nao retorna nada.
    """

    # Carrega o dataset como DataFrame (as_frame=True retorna pandas DataFrame)
    diabetes = load_diabetes(as_frame=True)

    # Copia o DataFrame para nao alterar o objeto original do sklearn
    df = diabetes.frame.copy()

    # O sklearn chama a variavel alvo de "target" — renomeamos para algo
    # mais descritivo, deixando claro o que estamos prevendo
    df = df.rename(columns={"target": "progression"})

    # Monta o caminho para data/raw.csv a partir da raiz do projeto
    out_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw.csv")
    )

    # Salva sem o indice do pandas (seria uma coluna extra desnecessaria)
    df.to_csv(out_path, index=False)
    print(f"Dataset salvo em: {out_path}")
    print(f"Shape: {df.shape}")  # ex: (442, 11)

    # Principio de Limpeza: deixa os dados prontos para o treino
    # Normaliza as features com StandardScaler e salva o scaler em disco
    from src.features.build_features import build
    build()


# So executa se chamado diretamente (nao quando importado)
if __name__ == "__main__":
    download()
