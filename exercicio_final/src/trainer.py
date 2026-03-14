"""
src/trainer.py — Exercicio Final: Penguins MLOps
=================================================
Responsabilidade: treinar dois modelos separados e salvar os artefatos.

    Modelo A (Classificacao): RandomForestClassifier → prediz a especie
    Modelo B (Regressao):     RandomForestRegressor  → estima o peso (g)

Por que dois modelos separados e nao MultiOutputRegressor?
    MultiOutputRegressor e mais elegante, mas exige que todas as saidas
    sejam numericas. Como a especie e uma categoria, usar dois modelos
    separados e mais claro, mais facil de depurar e mais flexivel para
    usar algoritmos diferentes em cada tarefa.

Principio de Reprodutibilidade:
    - Se os dados processados nao existirem, o trainer chama automaticamente
      o data_loader e o preprocessor antes de treinar.
    - Salva os modelos em models/ (pasta deletavel para recriar do zero).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Adiciona a raiz do projeto ao path para permitir "from src.x import y"
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_loader import TRAIN_CSV
from src.preprocessor import limpar_e_preparar

# Pasta onde os modelos serao salvos
_MODELS = os.path.join(ROOT, "models")
MODEL_CLASSIFIER = os.path.join(_MODELS, "classifier.joblib")
MODEL_REGRESSOR  = os.path.join(_MODELS, "regressor.joblib")

# Parametros dos modelos
N_ESTIMATORS  = 200   # numero de arvores na floresta (mais arvores = mais estavel)
RANDOM_STATE  = 42    # semente para reproducibilidade
CV_FOLDS      = 5     # numero de folds na validacao cruzada


def train():
    """
    Treina o classificador de especie e o regressor de peso.

    Fluxo:
        1. Verifica se os dados processados existem (se nao, baixa e prepara)
        2. Carrega train.csv e aplica o preprocessamento (modo_treino=True)
        3. Treina RandomForestClassifier para a especie (Tarefa A)
        4. Treina RandomForestRegressor para o peso (Tarefa B)
        5. Avalia ambos com validacao cruzada 5-fold
        6. Salva os modelos em models/

    Nao recebe parametros e nao retorna nada.
    """

    # --- Passo 1: Verificar/carregar dados de treino ---
    if not os.path.exists(TRAIN_CSV):
        # Principio de Reprodutibilidade: baixa e prepara automaticamente
        print("Dados de treino nao encontrados. Executando data_loader...")
        from src.data_loader import download_data, split_data
        download_data()
        split_data()

    print(f"Carregando dados de treino: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"  {len(df_train)} amostras de treino")

    # --- Passo 2: Preprocessamento ---
    # modo_treino=True: ajusta e salva todos os transformadores (scaler, encoders, imputadores)
    print("\nAplicando preprocessamento (modo treino)...")
    X_train, y_especie, y_peso = limpar_e_preparar(df_train, modo_treino=True)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Classes de especie: {np.unique(y_especie)} (0=Adelie, 1=Chinstrap, 2=Gentoo)")

    # --- Passo 3: Treinar classificador de especie (Tarefa A) ---
    print("\n[Tarefa A] Treinando classificador de especie...")
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,      # usa todos os nucleos do processador
    )

    # Validacao cruzada: divide o treino em 5 partes, treina em 4 e valida em 1, rotacionando
    # Metrica: accuracy (fracao de predicoes corretas)
    cv_acc = cross_val_score(classifier, X_train, y_especie, cv=CV_FOLDS, scoring="accuracy")
    print(f"  CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")

    # Treina no conjunto completo de treino apos a validacao
    classifier.fit(X_train, y_especie)

    # --- Passo 4: Treinar regressor de peso (Tarefa B) ---
    print("\n[Tarefa B] Treinando regressor de peso corporal...")
    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Metrica: neg_root_mean_squared_error (RMSE negativo, pois sklearn minimiza)
    cv_rmse = cross_val_score(
        regressor, X_train, y_peso,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    # Converte para positivo para exibicao (RMSE em gramas)
    print(f"  CV RMSE: {-cv_rmse.mean():.1f}g (+/- {cv_rmse.std():.1f}g)")

    regressor.fit(X_train, y_peso)

    # --- Passo 5: Salvar modelos ---
    os.makedirs(_MODELS, exist_ok=True)

    joblib.dump(classifier, MODEL_CLASSIFIER)
    print(f"\nClassificador salvo: {MODEL_CLASSIFIER}")

    joblib.dump(regressor, MODEL_REGRESSOR)
    print(f"Regressor salvo:     {MODEL_REGRESSOR}")
    print("\nTreinamento concluido!")


if __name__ == "__main__":
    train()
