"""
src/data_loader.py — Exercicio Final: Penguins MLOps
=====================================================
Responsabilidade: carregar o dataset bruto de pinguins, limpar linhas
invalidas e dividir em conjuntos de treino e teste.

O dado bruto e o Palmer Penguins Dataset, disponivel via sklearn-datasets
(seaborn) ou download manual. Aqui usamos o seaborn para simplificar.

Principio de Limpeza: este modulo entrega os dados prontos para o
preprocessor.py — sem valores nulos nas colunas alvo.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Caminho absoluto para a pasta raiz do projeto (um nivel acima de src/)
ROOT     = Path(__file__).parent.parent

# Caminhos dos arquivos de dados
RAW_CSV  = ROOT / "data" / "raw"       / "penguins.csv"
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
TEST_CSV  = ROOT / "data" / "processed" / "test.csv"

# Colunas que o modelo vai usar como entrada
FEATURE_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "sex", "island"]

# Colunas alvo: uma para classificacao, outra para regressao
TARGET_CLASS = "species"     # Tarefa A: classificar a especie
TARGET_REG   = "body_mass_g" # Tarefa B: estimar o peso em gramas


def download_data():
    """
    Baixa o dataset Palmer Penguins usando seaborn e salva em data/raw/penguins.csv.

    O seaborn inclui este dataset por padrao — nao requer API key ou conta.
    Se o arquivo ja existe, pula o download (idempotente).
    """
    if RAW_CSV.exists():
        # Arquivo ja existe, informa e retorna sem baixar novamente
        print(f"Dataset ja existe: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
        print(f"  {df.shape[0]} registros, {df.shape[1]} colunas")
        return

    # Importa seaborn so quando necessario (evita ImportError se nao instalado)
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn nao instalado. Execute: pip install seaborn")

    print("Baixando Palmer Penguins Dataset via seaborn...")

    # seaborn.load_dataset busca o CSV do repositorio online do seaborn
    df = sns.load_dataset("penguins")

    # Garante que a pasta data/raw/ existe antes de salvar
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Salva o CSV bruto sem alterar nada (principio: raw = dado original)
    df.to_csv(RAW_CSV, index=False)

    print(f"Dataset salvo: {RAW_CSV}")
    print(f"  {df.shape[0]} pinguins, {df.shape[1]} colunas")
    print(f"  Especies: {sorted(df['species'].unique())}")
    print(f"  Valores nulos: {df.isnull().sum().sum()} no total")


def split_data(test_size: float = 0.2, random_state: int = 42):
    """
    Le o CSV bruto, remove linhas com valores nulos nos alvos e divide em treino/teste.

    Por que remover linhas nulas nos alvos?
        Linhas sem especie ou sem peso nao podem ser usadas no treino nem na avaliacao.
        Features com valores nulos sao tratadas pelo preprocessor.py (imputacao).

    Parametros:
        test_size    (float): fracao dos dados para teste (padrao: 20%)
        random_state (int):  semente para reproducibilidade

    Retorna:
        train (DataFrame): dados de treino com todas as colunas
        test  (DataFrame): dados de teste com todas as colunas
    """

    if not RAW_CSV.exists():
        # Se o CSV bruto nao existe, chama download automaticamente
        print("Dataset nao encontrado. Baixando...")
        download_data()

    # Le o CSV bruto
    df = pd.read_csv(RAW_CSV)
    print(f"Dataset carregado: {df.shape[0]} registros")

    # Remove linhas onde os alvos sao nulos (nao servem nem para treino nem para teste)
    n_antes = len(df)
    df = df.dropna(subset=[TARGET_CLASS, TARGET_REG])
    n_removidos = n_antes - len(df)
    if n_removidos > 0:
        print(f"  Removidos {n_removidos} registros com alvos nulos")

    # Divide em treino e teste, mantendo proporcao de especies (stratify)
    # stratify garante que a proporcao de cada especie seja igual no treino e no teste
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_CLASS],  # mantém proporcao de especies
    )

    # Garante que a pasta processed/ existe
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Salva os conjuntos de treino e teste como CSV
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV,  index=False)

    print(f"  Treino: {len(train)} amostras -> {TRAIN_CSV}")
    print(f"  Teste:  {len(test)} amostras  -> {TEST_CSV}")

    return train, test


if __name__ == "__main__":
    # Permite executar diretamente: python src/data_loader.py
    download_data()
    split_data()
