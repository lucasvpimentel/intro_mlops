"""
src/features/build_features.py — Exercicio 03: Wine Classifier
===============================================================
Responsabilidade: normalizar as 13 features quimicas do vinho com
StandardScaler e salvar o scaler em disco.

Por que normalizar?
    O Wine Dataset tem features em escalas muito diferentes:
        - 'magnesium' varia de 70 a 162 (escala de centenas)
        - 'alcohol' varia de 11 a 15 (escala de dezenas)
        - 'hue' varia de 0.4 a 1.7 (escala de unidades)
    Sem normalizacao, features com valores maiores dominam o aprendizado.
    O StandardScaler coloca todas na mesma escala (media=0, desvio=1).

Por que salvar o scaler?
    Na inferencia (predict.py), os novos dados precisam ser transformados
    com a MESMA media e desvio aprendidos no treino. Sem isso, os valores
    entrariam no modelo em escala diferente e as predicoes seriam erradas.

Como executar diretamente:
    python src/features/build_features.py
"""

import sys    # encerrar com mensagem de erro
from pathlib import Path
import joblib # serializar o scaler em disco
import pandas as pd  # leitura do CSV e montagem do DataFrame normalizado

# StandardScaler: transforma cada feature para media=0 e desvio_padrao=1
from sklearn.preprocessing import StandardScaler

# Caminhos calculados relativos a este arquivo
ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data" / "raw.csv"        # CSV bruto de entrada
SCALER_PATH = ROOT / "data" / "scaler.joblib"  # scaler serializado de saida
PROC_PATH   = ROOT / "data" / "processed.csv"  # CSV normalizado de saida

# As 13 features quimicas — mesmos nomes do download_data.py
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]

# Coluna alvo — nao normalizamos o alvo, apenas as features
TARGET = "cultivar"


def build(save_processed: bool = True):
    """
    Ajusta o StandardScaler nas features e salva scaler e dataset normalizado.

    Parametros:
        save_processed (bool): se True (padrao), salva processed.csv.
                               Desative para usar apenas o scaler sem criar o CSV.

    Retorna:
        X_scaled (ndarray): array numpy com as 13 features normalizadas
        y        (ndarray): array numpy com os nomes dos cultivares (sem alteracao)
        scaler   (StandardScaler): scaler ja ajustado (media e desvio aprendidos)
    """

    # Verifica se o arquivo de entrada existe
    if not DATA_PATH.exists():
        print("raw.csv nao encontrado. Execute: python main.py download")
        sys.exit(1)

    # Le o dataset bruto
    df = pd.read_csv(DATA_PATH)

    # Separa as features do alvo
    X = df[FEATURES]   # DataFrame com 13 colunas numericas
    y = df[TARGET]     # Serie com os nomes dos cultivares ('class_0', etc.)

    # Cria e ajusta o scaler:
    # fit_transform() faz duas coisas de uma vez:
    #   fit:       aprende a media e desvio de cada coluna em X
    #   transform: aplica a normalizacao (subtrai media, divide por desvio)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # resultado e um array numpy

    # Salva o scaler treinado para reutilizar na inferencia
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em: {SCALER_PATH}")

    if save_processed:
        # Reconstroi DataFrame com os dados normalizados e os nomes originais das colunas
        df_proc = pd.DataFrame(X_scaled, columns=FEATURES)

        # Adiciona a coluna alvo sem normalizar (ela e categorica, nao numerica)
        df_proc[TARGET] = y.values  # .values: converte Series para array numpy

        # Salva o CSV normalizado para uso no treino e inspecao
        df_proc.to_csv(PROC_PATH, index=False)
        print(f"Dataset normalizado salvo em: {PROC_PATH}")

    # Retorna os tres artefatos para uso direto em codigo
    return X_scaled, y.values, scaler


if __name__ == "__main__":
    build()
