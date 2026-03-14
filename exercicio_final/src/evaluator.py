"""
src/evaluator.py — Exercicio Final: Penguins MLOps
===================================================
Responsabilidade: avaliar os dois modelos no conjunto de teste e
gerar graficos de desempenho.

Metricas:
    - Classificador: Accuracy, Precision, Recall, F1 (por especie)
                     Matriz de Confusao
    - Regressor:     RMSE, MAE, R² (R-squared)
                     Grafico de erro por especie

Os graficos sao salvos em reports/.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # backend sem janela grafica (compativel com servidores)
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    root_mean_squared_error, mean_absolute_error, r2_score,
)

# Adiciona raiz do projeto ao path
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_loader import TEST_CSV
from src.preprocessor import limpar_e_preparar, PATH_LE_SPECIES

# Caminhos dos artefatos
_MODELS   = os.path.join(ROOT, "models")
_REPORTS  = os.path.join(ROOT, "reports")

MODEL_CLASSIFIER = os.path.join(_MODELS, "classifier.joblib")
MODEL_REGRESSOR  = os.path.join(_MODELS, "regressor.joblib")


def evaluate():
    """
    Avalia os dois modelos no conjunto de teste e salva graficos em reports/.

    Fluxo:
        1. Carrega test.csv e aplica preprocessamento (modo inferencia)
        2. Avalia o classificador: accuracy, classification_report, matriz de confusao
        3. Avalia o regressor: RMSE, MAE, R2, grafico de erro por especie
        4. Salva graficos em reports/

    Nao recebe parametros e nao retorna nada.
    """

    # Verifica se os modelos existem
    for path in [MODEL_CLASSIFIER, MODEL_REGRESSOR]:
        if not os.path.exists(path):
            print(f"Modelo nao encontrado: {path}")
            print("Execute: python main.py train")
            sys.exit(1)

    # Carrega dados de teste
    if not os.path.exists(TEST_CSV):
        print(f"Dados de teste nao encontrados: {TEST_CSV}")
        print("Execute: python main.py download")
        sys.exit(1)

    print(f"Carregando dados de teste: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)
    print(f"  {len(df_test)} amostras de teste")

    # Preprocessamento em modo inferencia (modo_treino=False):
    # Carrega os transformadores salvos no treino e aplica a mesma transformacao.
    # Nota: no modo inferencia, y_especie e y_peso retornam None — lemos os alvos
    # diretamente do dataframe de teste, pois eles sao conhecidos na avaliacao.
    X_test, _, _ = limpar_e_preparar(df_test, modo_treino=False)

    # Extrai os alvos verdadeiros diretamente do CSV de teste
    y_peso_true        = df_test["body_mass_g"]      # peso real para avaliar o regressor

    # Carrega os modelos treinados
    classifier = joblib.load(MODEL_CLASSIFIER)
    regressor  = joblib.load(MODEL_REGRESSOR)

    # Carrega o encoder de especies para converter numeros de volta em nomes
    le_species = joblib.load(PATH_LE_SPECIES)

    # Converte especies de texto para numeros para comparacao com as predicoes
    y_especie_true_num = le_species.transform(df_test["species"])

    # --- Avaliacao do Classificador (Tarefa A) ---
    print("\n" + "=" * 60)
    print("  TAREFA A — Classificacao de Especie")
    print("=" * 60)

    # Predicoes do classificador
    y_especie_pred = classifier.predict(X_test)

    # Accuracy: fracao de predicoes corretas
    acc = accuracy_score(y_especie_true_num, y_especie_pred)
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Classification report: precision, recall e F1 por classe
    # Precision: dos que eu disse "Adelie", quantos eram realmente Adelie?
    # Recall:    de todos os Adelie reais, quantos eu acertei?
    # F1: media harmonica de precision e recall
    print("\n  Classification Report:")
    nomes_especies = le_species.classes_  # ["Adelie", "Chinstrap", "Gentoo"]
    print(classification_report(y_especie_true_num, y_especie_pred, target_names=nomes_especies))

    # Matriz de confusao: linhas = real, colunas = predito
    cm = confusion_matrix(y_especie_true_num, y_especie_pred)

    # --- Grafico: Matriz de Confusao ---
    os.makedirs(_REPORTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,          # mostra os numeros dentro das celulas
        fmt="d",             # formato inteiro
        cmap="Blues",        # escala de cores azul
        xticklabels=nomes_especies,
        yticklabels=nomes_especies,
        ax=ax,
    )
    ax.set_title("Matriz de Confusao — Classificador de Especie", fontsize=13)
    ax.set_xlabel("Especie Predita")
    ax.set_ylabel("Especie Real")
    plt.tight_layout()
    cm_path = os.path.join(_REPORTS, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Grafico salvo: {cm_path}")

    # --- Avaliacao do Regressor (Tarefa B) ---
    print("\n" + "=" * 60)
    print("  TAREFA B — Regressao de Peso Corporal")
    print("=" * 60)

    # Predicoes do regressor
    y_peso_pred = regressor.predict(X_test)

    # RMSE: raiz do erro quadratico medio (em gramas)
    # Penaliza mais os erros grandes
    rmse = root_mean_squared_error(y_peso_true, y_peso_pred)

    # MAE: erro absoluto medio (em gramas)
    # Mais intuitivo: "em media, erro X gramas"
    mae = mean_absolute_error(y_peso_true, y_peso_pred)

    # R²: quanto da variancia do peso o modelo explica (1.0 = perfeito, 0.0 = sem predicao)
    r2 = r2_score(y_peso_true, y_peso_pred)

    print(f"  RMSE : {rmse:.1f} g")
    print(f"  MAE  : {mae:.1f} g")
    print(f"  R²   : {r2:.4f}")

    # --- Grafico: Erro de peso por especie ---
    erros = np.abs(y_peso_true.values - y_peso_pred)
    especies_pred = le_species.inverse_transform(y_especie_pred)

    df_erros = pd.DataFrame({
        "especie": especies_pred,
        "erro_abs": erros,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_erros,
        x="especie",
        y="erro_abs",
        hue="especie",   # necessario a partir do seaborn 0.14 para usar palette com x
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Erro Absoluto de Peso por Especie", fontsize=13)
    ax.set_xlabel("Especie (predita pelo Classificador)")
    ax.set_ylabel("Erro Absoluto (g)")
    plt.tight_layout()
    err_path = os.path.join(_REPORTS, "weight_error_by_species.png")
    plt.savefig(err_path, dpi=150)
    plt.close()
    print(f"  Grafico salvo: {err_path}")

    print("\nAvaliacao concluida!")


if __name__ == "__main__":
    evaluate()
