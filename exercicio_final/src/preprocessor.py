"""
src/preprocessor.py — Exercicio Final: Penguins MLOps
======================================================
Responsabilidade: tratar valores ausentes, codificar variaveis categoricas
e normalizar features numericas.

Modo treino (modo_treino=True):
    - Ajusta (fit) todos os transformadores nos dados de treino
    - Salva os transformadores em models/ para uso na inferencia

Modo inferencia (modo_treino=False):
    - Carrega os transformadores salvos (nao refaz o fit!)
    - Aplica a mesma transformacao nos dados novos

Por que separar fit de transform?
    No treino, aprendemos "qual e a media?" ou "quais categorias existem?".
    Na inferencia, aplicamos o que foi aprendido. Refazer o fit na inferencia
    seria data leakage e causaria inconsistencia nos dados.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Caminhos dos artefatos salvos (relativos a raiz do projeto)
ROOT        = Path(__file__).parent.parent
MODELS_DIR  = ROOT / "models"

# Nomes dos arquivos de artefatos
PATH_LE_SEX     = MODELS_DIR / "le_sex.joblib"
PATH_LE_ISLAND  = MODELS_DIR / "le_island.joblib"
PATH_LE_SPECIES = MODELS_DIR / "le_species.joblib"
PATH_SCALER     = MODELS_DIR / "scaler.joblib"

# Definicao das colunas
FEATURE_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "sex", "island"]
COLS_NUM     = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]  # numericas
COLS_CAT     = ["sex", "island"]                                          # categoricas
TARGET_CLASS = "species"
TARGET_REG   = "body_mass_g"


def limpar_e_preparar(df: pd.DataFrame, modo_treino: bool = True):
    """
    Tratamento completo dos dados dos pinguins.

    Etapas aplicadas em ordem:
        1. Selecao das features relevantes
        2. Imputacao de valores ausentes (media para numericas, moda para categoricas)
        3. Codificacao de categorias em numeros (LabelEncoder)
        4. Normalizacao das features numericas (StandardScaler)

    Parametros:
        df          (DataFrame): dados brutos com colunas originais
        modo_treino (bool):      True = fit+transform e salva artefatos
                                 False = carrega artefatos e so aplica transform

    Retorna:
        X_final     (ndarray):  matriz de features processada, shape (n, 5)
        y_especie   (ndarray ou None): vetor de classes numericas (treino) ou None
        y_peso      (Series ou None):  vetor de pesos em gramas (treino) ou None
    """

    # Garante que a pasta models/ existe antes de salvar artefatos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Etapa 1: Selecao de features ---
    # Copia apenas as colunas que o modelo vai usar (evita modificar o df original)
    X = df[FEATURE_COLS].copy()

    # Extrai os alvos se estivermos em modo treino
    y_especie = df[TARGET_CLASS] if modo_treino else None
    y_peso    = df[TARGET_REG]   if modo_treino else None

    # --- Etapa 2: Imputacao de Valores Ausentes ---
    # SimpleImputer substitui NaN pela media (numericas) ou valor mais frequente (categoricas)
    imputer_num = SimpleImputer(strategy="mean")           # ex: bill_length_mm faltante → media
    imputer_cat = SimpleImputer(strategy="most_frequent")  # ex: sex faltante → categoria mais comum

    if modo_treino:
        # No treino: ajusta os imputadores nos dados de treino e transforma
        X[COLS_NUM] = imputer_num.fit_transform(X[COLS_NUM])
        X[COLS_CAT] = imputer_cat.fit_transform(X[COLS_CAT])

        # Salva os imputadores para uso na inferencia
        joblib.dump(imputer_num, MODELS_DIR / "imputer_num.joblib")
        joblib.dump(imputer_cat, MODELS_DIR / "imputer_cat.joblib")
    else:
        # Na inferencia: carrega os imputadores ajustados no treino
        imputer_num = joblib.load(MODELS_DIR / "imputer_num.joblib")
        imputer_cat = joblib.load(MODELS_DIR / "imputer_cat.joblib")

        # Aplica a mesma transformacao (usa os valores aprendidos no treino)
        X[COLS_NUM] = imputer_num.transform(X[COLS_NUM])
        X[COLS_CAT] = imputer_cat.transform(X[COLS_CAT])

    # --- Etapa 3: Codificacao de Variaveis Categoricas ---
    # LabelEncoder converte texto em numeros: ex. "male" → 1, "female" → 0
    # Importante: salvar o encoder para poder reverter a predicao na inferencia
    if modo_treino:
        # Cria encoders e os ajusta nos dados de treino
        le_sex     = LabelEncoder()
        le_island  = LabelEncoder()
        le_species = LabelEncoder()

        # Transforma as colunas categoricas em numeros
        X["sex"]    = le_sex.fit_transform(X["sex"])
        X["island"] = le_island.fit_transform(X["island"])

        # Codifica a variavel alvo (especie) em numeros para o classificador
        y_especie_num = le_species.fit_transform(y_especie)

        # Salva os encoders para decodificacao na inferencia
        joblib.dump(le_sex,     PATH_LE_SEX)
        joblib.dump(le_island,  PATH_LE_ISLAND)
        joblib.dump(le_species, PATH_LE_SPECIES)

    else:
        # Na inferencia: carrega os encoders do treino
        le_sex    = joblib.load(PATH_LE_SEX)
        le_island = joblib.load(PATH_LE_ISLAND)

        # Aplica a mesma codificacao (mesmos mapeamentos do treino)
        X["sex"]    = le_sex.transform(X["sex"])
        X["island"] = le_island.transform(X["island"])

        # Na inferencia nao temos o alvo de especie
        y_especie_num = None

    # --- Etapa 4: Normalizacao (StandardScaler) ---
    # StandardScaler subtrai a media e divide pelo desvio padrao de cada coluna.
    # Isso deixa todas as features na mesma escala (media=0, std=1),
    # o que melhora o desempenho de algoritmos sensivels a escala (SVM, KNN, etc.)
    # e geralmente ajuda RandomForest a convergir mais rapidamente.
    scaler = StandardScaler()

    if modo_treino:
        # No treino: aprende media e std de cada coluna e transforma
        X_final = scaler.fit_transform(X)

        # Salva o scaler para aplicar a mesma escala na inferencia
        joblib.dump(scaler, PATH_SCALER)
    else:
        # Na inferencia: carrega o scaler e aplica (sem aprender nada novo!)
        scaler  = joblib.load(PATH_SCALER)
        X_final = scaler.transform(X)

    return X_final, y_especie_num, y_peso


if __name__ == "__main__":
    print("Modulo preprocessor.py pronto para importacao.")
    print("Use: from src.preprocessor import limpar_e_preparar")
