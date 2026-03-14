"""
src/models/train.py — Exercicio 02: Diabetes Regressor
=======================================================
Responsabilidade: treinar um modelo de regressao no dataset Diabetes
normalizado e salvar o modelo treinado em disco.

Dois algoritmos disponiveis:
    - Ridge Regression (padrao): modelo linear com regularizacao L2.
      Bom para dados com features correlacionadas.
    - Random Forest Regressor: ensemble de arvores de decisao.
      Captura relacoes nao-lineares entre as features.

Principio de Reprodutibilidade:
    Se os dados processados nao existirem, este script baixa e prepara
    tudo automaticamente antes de treinar.

Como executar diretamente:
    python src/models/train.py                 # usa Ridge (padrao)
    python src/models/train.py --model rf      # usa Random Forest
"""

import sys      # adicionar raiz ao path de busca do Python
from pathlib import Path
import argparse # parser de argumentos da linha de comando
import joblib   # serializar o modelo em disco
import pandas as pd   # leitura do CSV
import numpy as np    # calculo do RMSE (raiz quadrada)

# Algoritmos de regressao do scikit-learn
from sklearn.linear_model import Ridge                   # Ridge: linear com regularizacao L2
from sklearn.ensemble import RandomForestRegressor       # Random Forest: nao-linear

# Utilitarios de avaliacao
from sklearn.model_selection import train_test_split, cross_val_score  # divisao e validacao cruzada
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # metricas de regressao

# Caminho raiz do projeto
ROOT = Path(__file__).parent.parent.parent

# Permite imports "from src...." ao rodar este arquivo diretamente
sys.path.insert(0, str(ROOT))

# Arquivo de entrada: dados ja normalizados pelo build_features.py
PROC_PATH  = ROOT / "data" / "processed.csv"

# Arquivo de saida: modelo serializado
MODEL_PATH = ROOT / "data" / "models" / "model.joblib"

# Features e alvo — devem ser exatamente os mesmos usados no build_features.py
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET   = "progression"


def build_model(model_type: str):
    """
    Fabrica de modelos: retorna o objeto do algoritmo escolhido.

    Parametros:
        model_type (str): 'ridge' para Ridge Regression, 'rf' para Random Forest

    Retorna:
        model: objeto do scikit-learn pronto para ser treinado com .fit()
    """
    if model_type == "ridge":
        # alpha=1.0: forca da regularizacao L2 (penaliza coeficientes grandes)
        return Ridge(alpha=1.0)

    # n_estimators=200: numero de arvores na floresta
    # random_state=42: garante reproducibilidade
    # n_jobs=-1: usa todos os nucleos do processador em paralelo
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def evaluate(name, model, X_test, y_test):
    """
    Calcula e imprime as metricas de avaliacao no conjunto de teste.

    Metricas usadas:
        RMSE (Root Mean Squared Error): erro medio em unidades da variavel alvo.
              Penaliza mais erros grandes. Quanto menor, melhor.
        MAE  (Mean Absolute Error): erro absoluto medio. Mais robusto a outliers.
        R2   (Coeficiente de Determinacao): quanto da variancia o modelo explica.
              1.0 = perfeito | 0.0 = igual a prever a media | negativo = pior que a media.

    Parametros:
        name   (str):     nome do modelo para exibir
        model:            modelo ja treinado com .predict()
        X_test (ndarray): features do conjunto de teste
        y_test (ndarray): valores reais do conjunto de teste
    """
    y_pred = model.predict(X_test)  # gera predicoes para o conjunto de teste

    # Calcula cada metrica
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # raiz do erro quadratico medio
    mae  = mean_absolute_error(y_test, y_pred)           # erro absoluto medio
    r2   = r2_score(y_test, y_pred)                      # coeficiente de determinacao

    print(f"\n=== {name} — Conjunto de Teste ===")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R2   : {r2:.4f}")


def train(model_type: str = "ridge"):
    """
    Executa o pipeline completo de treinamento:
        1. Garante que os dados processados existem
        2. Le o CSV normalizado
        3. Divide em treino (80%) e teste (20%)
        4. Avalia com validacao cruzada 5-fold no treino
        5. Treina o modelo escolhido
        6. Avalia no conjunto de teste (RMSE, MAE, R2)
        7. Salva o modelo em data/models/model.joblib

    Parametros:
        model_type (str): 'ridge' (padrao) ou 'rf'
    """

    # Principio de Reprodutibilidade: garante os dados antes de treinar
    # download() ja inclui a normalizacao (build_features), entao um unico
    # ponto de entrada cobre todo o pipeline de dados
    if not PROC_PATH.exists():
        print("processed.csv nao encontrado. Preparando dados automaticamente...")
        from src.data.download_data import download
        download()

    # Cria a pasta data/models/ se nao existir
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Carrega os dados ja normalizados
    df = pd.read_csv(PROC_PATH)

    # .values converte DataFrame/Series para array numpy (formato esperado pelo sklearn)
    X = df[FEATURES].values
    y = df[TARGET].values

    # Divide em treino e teste com a mesma semente para reproducibilidade
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instancia o modelo de acordo com o tipo escolhido
    model = build_model(model_type)

    # Validacao Cruzada (Cross-Validation) com 5 folds:
    # Divide o treino em 5 partes; treina em 4 e valida na 5a; repete 5 vezes.
    # Isso da uma estimativa mais confiavel da performance antes do teste final.
    # n_jobs=-1: paraleliza os 5 folds usando todos os nucleos
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print(f"CV R2 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Treina o modelo final com todos os dados de treino
    model.fit(X_train, y_train)

    # Avalia no conjunto de teste (dados que o modelo nunca viu)
    evaluate(model_type.upper(), model, X_test, y_test)

    # Salva o modelo em disco
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")


# Ponto de entrada com suporte a argumento --model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --model e opcional: se nao fornecido, usa 'ridge' como padrao
    parser.add_argument("--model", choices=["ridge", "rf"], default="ridge")
    args = parser.parse_args()
    train(args.model)
