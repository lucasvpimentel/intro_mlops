"""
src/models/predict.py — Exercicio 03: Wine Classifier
======================================================
Responsabilidade: inferencia em lote (batch inference).

Le data/input.json com uma lista de vinhos, aplica o scaler do treino,
classifica cada um com o modelo salvo e gera data/output.csv.

Principio de Isolamento:
    Este modulo NAO sabe como o modelo foi treinado nem como o scaler
    foi ajustado. Ele apenas carrega os arquivos .joblib e usa as
    interfaces padrao do scikit-learn (.transform, .predict, .predict_proba).

Formato esperado do input.json:
    [
      {"alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, ...},
      {"alcohol": 14.1, "malic_acid": 2.01, "ash": 2.30, ...}
    ]
    Cada objeto deve ter exatamente as 13 features do modelo.

Como executar diretamente:
    python src/models/predict.py
"""

import os     # caminhos de arquivo
import sys    # encerrar com erro se artefatos nao encontrados
import json   # leitura do arquivo JSON de entrada
import joblib # carregar scaler e modelo do disco
import pandas as pd  # criar DataFrame com nomes de coluna

# Caminhos raiz e dos artefatos
ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCALER_PATH = os.path.join(ROOT, "data", "scaler.joblib")              # gerado por build_features.py
MODEL_PATH  = os.path.join(ROOT, "data", "models", "wine_model.joblib") # gerado por train.py
INPUT_PATH  = os.path.join(ROOT, "data", "input.json")                 # arquivo de entrada padrao
OUTPUT_PATH = os.path.join(ROOT, "data", "output.csv")                 # arquivo de saida padrao

# As 13 features — mesma ordem usada no treino (a ordem importa para o scaler)
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]


def load_artifacts():
    """
    Carrega o scaler e o modelo do disco.

    Verifica se ambos existem. Se algum estiver faltando, exibe
    mensagem orientadora e encerra o programa.

    Retorna:
        scaler: StandardScaler com media e desvio do treino
        model:  RandomForestClassifier treinado
    """

    # Checa quais arquivos estao ausentes
    missing = [p for p in [SCALER_PATH, MODEL_PATH] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Nao encontrado: {p}")
        print("Execute: python main.py pipeline")
        sys.exit(1)

    # Carrega e retorna os dois objetos salvos em disco
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)


def predict_batch(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    """
    Le o arquivo JSON de entrada, classifica cada vinho e salva o resultado em CSV.

    Fluxo:
        1. Le input.json e cria um DataFrame
        2. Valida que todas as features necessarias estao presentes
        3. Aplica o scaler (mesma transformacao do treino)
        4. Gera as predicoes e probabilidades com o modelo
        5. Monta o DataFrame de saida com cultivar_previsto e confianca_pct
        6. Salva output.csv

    Parametros:
        input_path  (str): caminho do JSON de entrada (padrao: data/input.json)
        output_path (str): caminho do CSV de saida (padrao: data/output.csv)

    Retorna:
        df_output (DataFrame): tabela com as features + predicoes + confianca
    """

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_path):
        print(f"Arquivo de entrada nao encontrado: {input_path}")
        sys.exit(1)

    # Le o JSON — deve ser uma lista de objetos (cada objeto = um vinho)
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)  # converte JSON para lista de dicionarios Python

    # Converte a lista de dicionarios para um DataFrame do pandas
    df_input = pd.DataFrame(records)

    # Verifica se todas as 13 features estao presentes no JSON
    missing_cols = [c for c in FEATURES if c not in df_input.columns]
    if missing_cols:
        print(f"Colunas faltando no input.json: {missing_cols}")
        sys.exit(1)

    # Carrega o scaler e o modelo do disco
    scaler, model = load_artifacts()

    # Aplica a MESMA normalizacao do treino
    # scaler.transform() usa media e desvio aprendidos — nao re-ajusta
    X_scaled = scaler.transform(df_input[FEATURES])

    # Gera as predicoes de classe para cada vinho
    # Ex: ['class_0', 'class_1', 'class_0', ...]
    classes = model.predict(X_scaled)

    # predict_proba() retorna a probabilidade de cada classe para cada amostra
    # .max(axis=1) pega a maior probabilidade de cada linha (a classe prevista)
    # * 100 converte de proporcao (0-1) para percentual (0-100)
    probas     = model.predict_proba(X_scaled)
    confidence = probas.max(axis=1) * 100

    # Monta o DataFrame de saida com as features originais + predicoes
    df_output = df_input[FEATURES].copy()                    # copia as features de entrada
    df_output["cultivar_previsto"] = classes                 # adiciona a classe prevista
    df_output["confianca_pct"]     = confidence.round(1)     # adiciona a confianca (1 casa decimal)

    # Salva o CSV de saida
    df_output.to_csv(output_path, index=False)

    # Exibe o resultado formatado com o mesmo estilo do restante do projeto
    print("\n" + "=" * 55)
    print("  PREDICAO EM LOTE — Wine Classifier")
    print("=" * 55)
    print(f"  Vinhos processados : {len(df_output)}")
    print(f"  Resultado salvo em : {output_path}")
    print("-" * 55)
    print(df_output[["cultivar_previsto", "confianca_pct"]].to_string())
    print("=" * 55)

    return df_output


if __name__ == "__main__":
    predict_batch()
