"""
src/inference.py — Exercicio Final: Penguins MLOps
===================================================
Responsabilidade: carregar os modelos treinados e realizar predicoes
em novos dados sem precisar reprocessar ou retreinar.

Este e o script de USO DIARIO da equipe de biologos:
    - Recebe medicoes de campo (bico, asas, sexo, ilha)
    - Retorna especie predita + confianca
    - Retorna estimativa de peso corporal

Principio de Isolamento:
    Este modulo SO le artefatos .joblib — nunca acessa dados de treino.
    O preprocessamento usa modo_treino=False, carregando os transformadores
    que foram salvos durante o treino.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Adiciona raiz do projeto ao path
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.preprocessor import limpar_e_preparar, PATH_LE_SPECIES, FEATURE_COLS

# Caminhos dos modelos
_MODELS          = os.path.join(ROOT, "models")
MODEL_CLASSIFIER = os.path.join(_MODELS, "classifier.joblib")
MODEL_REGRESSOR  = os.path.join(_MODELS, "regressor.joblib")

# Caminho do arquivo de amostras para inferencia em lote
SAMPLES_PATH = os.path.join(ROOT, "data", "samples", "new_penguins.json")


def predict_single(bill_length_mm: float, bill_depth_mm: float,
                   flipper_length_mm: float, sex: str, island: str) -> dict:
    """
    Prediz especie e peso de um unico pinguim a partir de medicoes de campo.

    Parametros:
        bill_length_mm    (float): comprimento do bico em mm
        bill_depth_mm     (float): profundidade do bico em mm
        flipper_length_mm (float): comprimento da asa em mm
        sex               (str):   "male" ou "female"
        island            (str):   "Biscoe", "Dream" ou "Torgersen"

    Retorna:
        dict com:
            especie          (str):   especie predita ("Adelie", "Chinstrap" ou "Gentoo")
            confianca_pct    (float): confianca do classificador em %
            peso_estimado_g  (float): estimativa de peso corporal em gramas
    """

    # Verifica se os modelos existem antes de tentar carregar
    for path in [MODEL_CLASSIFIER, MODEL_REGRESSOR]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo nao encontrado: {path}\n"
                "Execute: python main.py train"
            )

    # Normaliza sex e island para corresponder ao formato do dataset de treino
    # O seaborn penguins usa "Male"/"Female" e "Biscoe"/"Dream"/"Torgersen" (capitalizados)
    sex    = sex.capitalize()     # "male" → "Male", "female" → "Female"
    island = island.capitalize()  # "biscoe" → "Biscoe", etc.

    # Cria um DataFrame com uma linha (o preprocessador espera um DataFrame)
    # As colunas devem ter os mesmos nomes usados no treino
    df = pd.DataFrame([{
        "bill_length_mm":    bill_length_mm,
        "bill_depth_mm":     bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "sex":               sex,
        "island":            island,
    }])

    # Aplica o mesmo preprocessamento do treino (modo_treino=False)
    # Isso garante que as features estejam na mesma escala que o modelo espera
    X, _, _ = limpar_e_preparar(df, modo_treino=False)

    # Carrega os modelos (leitura rapida do arquivo .joblib)
    classifier = joblib.load(MODEL_CLASSIFIER)
    regressor  = joblib.load(MODEL_REGRESSOR)
    le_species = joblib.load(PATH_LE_SPECIES)

    # --- Predicao de Especie (Tarefa A) ---
    # predict_proba retorna probabilidades para cada classe
    proba          = classifier.predict_proba(X)[0]  # [0] pega a primeira (unica) amostra
    classe_num     = int(np.argmax(proba))            # indice da classe com maior probabilidade
    especie        = le_species.inverse_transform([classe_num])[0]  # converte numero → nome
    confianca_pct  = round(float(proba[classe_num]) * 100, 1)       # em porcentagem

    # --- Predicao de Peso (Tarefa B) ---
    peso_estimado = round(float(regressor.predict(X)[0]), 1)

    return {
        "especie":         especie,
        "confianca_pct":   confianca_pct,
        "peso_estimado_g": peso_estimado,
    }


def predict_batch(samples_path: str = SAMPLES_PATH) -> list:
    """
    Realiza predicoes em lote a partir de um arquivo JSON.

    Formato esperado do JSON (lista de dicionarios):
    [
        {"bill_length_mm": 39.1, "bill_depth_mm": 18.7,
         "flipper_length_mm": 181.0, "sex": "male", "island": "Torgersen"},
        ...
    ]

    Parametros:
        samples_path (str): caminho para o arquivo JSON com as amostras

    Retorna:
        lista de dicts com os resultados de cada pinguim
    """

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Arquivo de amostras nao encontrado: {samples_path}")

    # Le o JSON com as amostras
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("\n" + "=" * 55)
    print("  PREDICAO EM LOTE — Penguins MLOps")
    print("=" * 55)
    print(f"  Arquivo : {samples_path}")
    print(f"  Total   : {len(samples)} pinguins")
    print("-" * 55)
    print(f"  {'#':>2}  {'Especie':<12} {'Confianca':>9}  {'Peso (g)':>8}")
    print(f"  {'-'*2}  {'-'*12} {'-'*9}  {'-'*8}")

    results = []
    for i, sample in enumerate(samples):
        # Chama predict_single para cada pinguim
        result = predict_single(
            bill_length_mm    = sample["bill_length_mm"],
            bill_depth_mm     = sample["bill_depth_mm"],
            flipper_length_mm = sample["flipper_length_mm"],
            sex               = sample["sex"],
            island            = sample["island"],
        )
        # Adiciona os dados de entrada ao resultado para facilitar comparacao
        result["entrada"] = sample
        results.append(result)

        print(
            f"  [{i+1:2d}]  {result['especie']:<12} "
            f"{result['confianca_pct']:>8.1f}%  "
            f"{result['peso_estimado_g']:>7.0f}g"
        )

    print("=" * 55)
    return results


if __name__ == "__main__":
    # Exemplo de uso direto: python src/inference.py
    print("Testando predicao individual...")
    resultado = predict_single(
        bill_length_mm=39.1,
        bill_depth_mm=18.7,
        flipper_length_mm=181.0,
        sex="male",
        island="Torgersen",
    )
    print(f"\nResultado: {resultado}")
