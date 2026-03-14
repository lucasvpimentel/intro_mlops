"""
src/models/train.py — Exercicio 03: Wine Classifier
====================================================
Responsabilidade: treinar um Random Forest Classifier no Wine Dataset
normalizado e salvar o modelo em disco.

Por que Random Forest?
    O dataset tem 13 features com correlacoes entre si (multicolinearidade).
    Random Forest e robusto a isso pois cada arvore usa um subconjunto
    aleatorio de features, reduzindo o impacto da correlacao.

Principio de Reprodutibilidade:
    Se os dados processados nao existirem, este script baixa e prepara
    tudo automaticamente antes de treinar. Basta rodar este arquivo.

Como executar diretamente:
    python src/models/train.py
"""

import sys    # adicionar raiz ao path de busca
from pathlib import Path
import joblib # serializar o modelo
import pandas as pd  # leitura do CSV normalizado

# Random Forest: ensemble de multiplas arvores de decisao
from sklearn.ensemble import RandomForestClassifier

# Utilitarios de avaliacao e divisao de dados
from sklearn.model_selection import train_test_split, cross_val_score

# Caminho raiz do projeto
ROOT = Path(__file__).parent.parent.parent

# Permite imports "from src...." ao rodar este arquivo diretamente
sys.path.insert(0, str(ROOT))

# Arquivo de entrada: CSV normalizado pelo build_features.py
PROC_PATH  = ROOT / "data" / "processed.csv"

# Arquivo de saida: modelo serializado
MODEL_PATH = ROOT / "data" / "models" / "wine_model.joblib"

# As 13 features — mesmos nomes e mesma ordem do build_features.py
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]

# Coluna alvo: nome do cultivar ('class_0', 'class_1', 'class_2')
TARGET = "cultivar"


def train():
    """
    Executa o pipeline completo de treinamento:
        1. Garante que os dados processados existem
        2. Le o CSV normalizado
        3. Divide em treino (80%) e teste (20%) com estratificacao
        4. Avalia com validacao cruzada 5-fold
        5. Treina o Random Forest com todos os dados de treino
        6. Salva o modelo em data/models/wine_model.joblib

    Nao recebe parametros.

    Retorna:
        model  : modelo treinado
        X_test : features do conjunto de teste (para uso externo, ex: evaluate.py)
        y_test : classes reais do conjunto de teste
    """

    # Principio de Reprodutibilidade: baixa e prepara os dados se necessario
    # download() ja inclui a normalizacao, entao um unico ponto de entrada e suficiente
    if not PROC_PATH.exists():
        print("processed.csv nao encontrado. Preparando dados automaticamente...")
        from src.data.download_data import download
        download()

    # Cria data/models/ se nao existir ainda
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Carrega o dataset normalizado
    df = pd.read_csv(PROC_PATH)

    # .values: converte para array numpy — formato esperado pelo sklearn
    X = df[FEATURES].values
    y = df[TARGET].values

    # Divide em treino (80%) e teste (20%)
    # stratify=y: garante proporcao igual das 3 classes em treino e teste
    # random_state=42: divisao sempre igual para reproducibilidade
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cria o modelo Random Forest
    # n_estimators=200: usa 200 arvores de decisao no ensemble
    # random_state=42: garante que as arvores aleatorias sejam as mesmas sempre
    # n_jobs=-1: usa todos os nucleos do processador para treinar em paralelo
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    # Validacao cruzada com 5 folds — avalia antes do treino final
    # Cada fold usa 80% para treinar e 20% para validar, rotacionando
    # scoring="accuracy": metrica usada e a acuracia (% de acertos)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Treina o modelo final com todos os dados de treino
    model.fit(X_train, y_train)

    # Salva o modelo em disco
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

    # Retorna modelo e dados de teste para o evaluate.py usar
    return model, X_test, y_test


if __name__ == "__main__":
    train()
