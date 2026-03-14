"""
src/features/build_features.py — Exercicio 02: Diabetes Regressor
==================================================================
Responsabilidade: normalizar as features com StandardScaler e salvar
o scaler em disco para uso posterior na inferencia.

Por que salvar o scaler separadamente do modelo?
    Durante o treino, o StandardScaler aprende a media e o desvio padrao
    de cada feature. Na inferencia, os dados novos precisam ser transformados
    com EXATAMENTE os mesmos valores aprendidos. Se usassemos um scaler novo
    na inferencia, os dados entrariam no modelo em escala diferente e as
    predicoes seriam incorretas.

    Solucao: salvar o scaler ajustado como scaler.joblib e reutiliza-lo
    sempre que novos dados precisarem ser transformados.

Como executar diretamente:
    python src/features/build_features.py
"""

import sys    # encerrar com erro se arquivo nao encontrado
from pathlib import Path
import joblib # salvar e carregar objetos Python em disco
import pandas as pd  # leitura do CSV e criacao de DataFrame normalizado

# StandardScaler: transforma cada feature para ter media=0 e desvio_padrao=1
# Isso evita que features com escalas maiores dominem o modelo
from sklearn.preprocessing import StandardScaler

# Caminhos calculados em relacao a este arquivo
ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data" / "raw.csv"        # entrada: dados brutos
SCALER_PATH = ROOT / "data" / "scaler.joblib"  # saida: scaler serializado
PROC_PATH   = ROOT / "data" / "processed.csv"  # saida: dados normalizados

# As 10 features clinicas — exatamente as colunas que o scaler vai transformar
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# Nome da coluna alvo — nao normalizamos o alvo, apenas as features
TARGET = "progression"


def build(save_processed: bool = True):
    """
    Ajusta o StandardScaler nas features de treino e salva tudo em disco.

    Parametros:
        save_processed (bool): se True (padrao), salva o dataset normalizado
                               como processed.csv. Util para inspecionar os
                               valores apos a normalizacao.

    Retorna:
        X_scaled (ndarray): matriz numpy com as features normalizadas
        y        (ndarray): array numpy com os valores alvo (sem alteracao)
        scaler   (StandardScaler): o scaler ja ajustado (media e desvio aprendidos)
    """

    # Verifica se o CSV bruto existe antes de tentar le-lo
    if not DATA_PATH.exists():
        print("raw.csv nao encontrado. Execute primeiro:")
        print("  python src/data/download_data.py")
        sys.exit(1)  # encerra com codigo de erro

    # Carrega o CSV com todos os dados
    df = pd.read_csv(DATA_PATH)

    # Separa as colunas de entrada (features) da coluna alvo
    X = df[FEATURES]   # DataFrame com as 10 features clinicas
    y = df[TARGET]     # Serie com os valores de progressao

    # Cria o objeto StandardScaler
    # Ao chamar fit_transform(), ele:
    #   1. Aprende a media e o desvio padrao de cada coluna (fit)
    #   2. Subtrai a media e divide pelo desvio de cada valor (transform)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # retorna um array numpy

    # Salva o scaler em disco para reutilizar na inferencia
    # joblib e mais eficiente que pickle para arrays numpy grandes
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em: {SCALER_PATH}")

    if save_processed:
        # Reconstroi um DataFrame com os dados normalizados e os nomes de coluna
        df_proc = pd.DataFrame(X_scaled, columns=FEATURES)

        # Adiciona a coluna alvo de volta (sem normalizar)
        df_proc[TARGET] = y.values  # .values converte Series para array numpy

        # Salva o dataset normalizado para inspecao e uso no treino
        df_proc.to_csv(PROC_PATH, index=False)
        print(f"Dataset normalizado salvo em: {PROC_PATH}")

    # Retorna os dados para quem chamar esta funcao diretamente no codigo
    return X_scaled, y.values, scaler


# So executa se chamado diretamente
if __name__ == "__main__":
    build()
