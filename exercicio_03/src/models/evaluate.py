"""
src/models/evaluate.py — Exercicio 03: Wine Classifier
=======================================================
Responsabilidade: carregar o modelo treinado, avalia-lo no conjunto
de teste e salvar o relatorio em data/evaluation.txt.

Por que ter um modulo de avaliacao separado do treino?
    - Permite re-avaliar sem re-treinar (util se voce mudar a metrica)
    - O pipeline (run_pipeline.bat) chama avaliacao como etapa propria
    - Mantem o train.py focado apenas em treinar

Principio de Isolamento (parcial):
    Este modulo carrega o modelo via joblib, sem importar o train.py.
    Ele recria o mesmo split de teste (mesma semente) para garantir
    que avalia exatamente os mesmos dados que nao foram vistos no treino.

Como executar diretamente:
    python src/models/evaluate.py
"""

import os     # caminhos de arquivo
import joblib # carregar o modelo do disco
import pandas as pd  # leitura do CSV
import numpy as np   # operacoes numericas (necessario para o split)

# Metricas e utilitarios do scikit-learn
from sklearn.model_selection import train_test_split  # recria o mesmo split do treino
from sklearn.metrics import (
    classification_report,  # tabela com precisao, recall e f1 por classe
    confusion_matrix,        # matriz mostrando acertos e erros por classe
    accuracy_score,          # percentual geral de acertos
)

# Caminhos calculados relativos a este arquivo
ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC_PATH  = os.path.join(ROOT, "data", "processed.csv")             # CSV normalizado
MODEL_PATH = os.path.join(ROOT, "data", "models", "wine_model.joblib") # modelo serializado
EVAL_PATH  = os.path.join(ROOT, "data", "evaluation.txt")            # relatorio de saida

# Features e alvo — iguais aos usados no treino
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]
TARGET = "cultivar"


def evaluate():
    """
    Avalia o modelo salvo no conjunto de teste e gera um relatorio.

    Estrategia:
        Usa o mesmo random_state=42 e test_size=0.2 do treino para
        garantir que o conjunto de teste aqui e identico ao do treino.
        Isso e essencial: avaliar no conjunto de teste correto.

    Saidas:
        - Imprime o relatorio no terminal
        - Salva o relatorio em data/evaluation.txt

    Nao recebe parametros e nao retorna nada.
    """

    # Verifica se o modelo existe antes de tentar carrega-lo
    if not os.path.exists(MODEL_PATH):
        print("Modelo nao encontrado. Execute: python main.py train")
        return  # encerra a funcao sem erro critico

    # Carrega o modelo serializado do disco
    model = joblib.load(MODEL_PATH)

    # Carrega os dados normalizados
    df = pd.read_csv(PROC_PATH)
    X  = df[FEATURES].values  # array numpy com as features
    y  = df[TARGET].values    # array numpy com os cultivares

    # Recria o MESMO split usado no treino
    # random_state=42 e stratify=y garantem a mesma divisao
    # Usamos _ para descartar X_train e y_train (nao precisamos deles aqui)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gera predicoes para o conjunto de teste
    y_pred = model.predict(X_test)

    # Calcula as metricas de avaliacao
    report = classification_report(y_test, y_pred)  # tabela por classe
    cm     = confusion_matrix(y_test, y_pred)        # matriz de confusao
    acc    = accuracy_score(y_test, y_pred)           # acuracia geral

    # Monta o relatorio como texto formatado
    lines = [
        "=== Avaliacao do Modelo — Wine Dataset ===\n",
        f"Accuracy: {acc:.4f}\n\n",       # acuracia geral (ex: 1.0000 = 100%)
        "Relatorio de Classificacao:\n",
        report,                            # precisao/recall/f1 por classe
        "\nMatriz de Confusao:\n",
        str(cm),                           # linhas = real, colunas = previsto
        "\n",
    ]
    output = "".join(lines)

    # Exibe no terminal
    print(output)

    # Salva em arquivo para registro permanente
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Relatorio salvo em: {EVAL_PATH}")


if __name__ == "__main__":
    evaluate()
