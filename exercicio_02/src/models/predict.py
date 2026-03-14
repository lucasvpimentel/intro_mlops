"""
src/models/predict.py — Exercicio 02: Diabetes Regressor
=========================================================
Responsabilidade: carregar o scaler e o modelo treinados e executar
a inferencia para novos pacientes.

Principio de Isolamento:
    Este modulo NAO sabe como o modelo foi treinado nem como o scaler
    foi ajustado. Ele apenas carrega os arquivos .joblib gerados pelos
    outros modulos e usa a interface padrao do scikit-learn.

    Isso significa que voce pode trocar o algoritmo de treino (ex: de
    Ridge para Random Forest) sem alterar nada neste arquivo.

Fluxo de inferencia:
    dados brutos do usuario
        -> normalizar com o mesmo scaler do treino  (scaler.transform)
        -> passar ao modelo                          (model.predict)
        -> retornar o valor previsto
"""

import os     # caminhos de arquivo
import sys    # encerrar com erro se artefatos nao encontrados
import joblib # carregar os objetos salvos em disco
import pandas as pd  # criar DataFrame com nomes de coluna para evitar warnings

# Caminho raiz do projeto
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Caminhos dos artefatos gerados pelo pipeline de treino
SCALER_PATH = os.path.join(ROOT, "data", "scaler.joblib")          # gerado por build_features.py
MODEL_PATH  = os.path.join(ROOT, "data", "models", "model.joblib") # gerado por train.py

# Mesma lista de features usada no treino — a ordem importa!
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def load_artifacts():
    """
    Carrega o scaler e o modelo do disco.

    Verifica se ambos os arquivos existem antes de tentar carrega-los.
    Se algum estiver faltando, exibe mensagem de ajuda e encerra.

    Retorna:
        scaler: objeto StandardScaler com media e desvio aprendidos no treino
        model:  objeto do sklearn (Ridge ou RandomForest) pronto para .predict()
    """

    # Verifica quais arquivos estao faltando
    missing = [p for p in [SCALER_PATH, MODEL_PATH] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Nao encontrado: {p}")
        print("Execute: python main.py pipeline")
        sys.exit(1)

    # Carrega e retorna os dois objetos
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)


def predict(values: list[float]) -> float:
    """
    Recebe os valores clinicos de um paciente e retorna a progressao estimada.

    Parametros:
        values (list[float]): lista com 10 valores nas features:
                              [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]

    Retorna:
        float: indice de progressao estimado (escala ~25 a ~346)
    """

    # Carrega os artefatos do disco
    scaler, model = load_artifacts()

    # Cria um DataFrame de uma linha com os nomes das colunas
    # Usar DataFrame (ao inves de array puro) garante que o sklearn
    # reconheca as features pelo nome e nao emita warnings
    df_input = pd.DataFrame([values], columns=FEATURES)

    # Aplica a MESMA transformacao usada no treino
    # scaler.transform() usa a media e desvio aprendidos no fit_transform()
    # sem re-aprender — esse e o ponto critico do principio de limpeza
    X_scaled = scaler.transform(df_input)

    # Retorna o valor previsto (primeiro elemento do array de predicoes)
    return model.predict(X_scaled)[0]


if __name__ == "__main__":
    # Exemplo de uso direto: python src/models/predict.py
    # Valores na escala pre-normalizada do sklearn diabetes dataset
    valores_exemplo = [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]

    resultado = predict(valores_exemplo)

    print("\n" + "=" * 50)
    print("  PREDICAO — Diabetes Regressor")
    print("=" * 50)
    print(f"  Entrada             : {valores_exemplo}")
    print(f"  Progressao (1 ano)  : {resultado:.1f}")
    print(f"  Escala: ~25 = baixa progressao | ~346 = alta")
    print("=" * 50)
