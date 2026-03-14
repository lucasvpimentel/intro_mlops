"""
src/data/download_data.py — Exercicio 01: Iris Classifier
==========================================================
Responsabilidade: baixar o dataset Iris e salvar em disco como CSV.

Principio de Limpeza:
    Ao final deste script, os dados ja estao prontos para o treino.
    O script de treino (src/models/train.py) pode ser executado
    imediatamente apos este sem nenhuma etapa manual.

Como executar diretamente:
    python src/data/download_data.py
"""

# Biblioteca do scikit-learn que contem datasets classicos de ML
from sklearn.datasets import load_iris

# pandas: biblioteca para manipulacao de tabelas de dados (DataFrames)
import pandas as pd

# os: modulo para trabalhar com caminhos de arquivos e diretorios
import os


def download():
    """
    Carrega o dataset Iris do scikit-learn e salva como CSV em data/raw.csv.

    O Iris Dataset contem 150 amostras de flores com 4 medidas cada:
        - sepal_length: comprimento da sepala (cm)
        - sepal_width:  largura da sepala (cm)
        - petal_length: comprimento da petala (cm)
        - petal_width:  largura da petala (cm)
    E o alvo (coluna 'species') com 3 classes: setosa, versicolor, virginica.

    Nao recebe parametros e nao retorna nada — apenas salva o arquivo.
    """

    # Carrega o dataset como um DataFrame do pandas (as_frame=True)
    # O resultado e um objeto com .frame (tabela completa), .target_names (nomes das classes), etc.
    iris = load_iris(as_frame=True)

    # Faz uma copia do DataFrame para nao modificar o objeto original
    df = iris.frame.copy()

    # O sklearn nomeia as colunas com nomes longos (ex: "sepal length (cm)")
    # Renomeamos para nomes mais simples e sem espacos, facilitando o uso no codigo
    df.columns = [
        "sepal_length",  # comprimento da sepala
        "sepal_width",   # largura da sepala
        "petal_length",  # comprimento da petala
        "petal_width",   # largura da petala
        "target",        # numero da classe: 0, 1 ou 2
    ]

    # Adiciona uma coluna com o nome legivel da especie
    # iris.target_names e um array: ['setosa', 'versicolor', 'virginica']
    # dict(enumerate(...)) cria: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    # .map() substitui cada numero pelo nome correspondente
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    # Monta o caminho absoluto para data/raw.csv
    # __file__ e o caminho deste proprio arquivo (download_data.py)
    # Subindo dois niveis ("..","..") chegamos na raiz do projeto
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw.csv"
    )
    # normpath remove redundancias como ".." do caminho final
    out_path = os.path.normpath(out_path)

    # Salva o DataFrame como CSV sem incluir o indice numerico do pandas
    df.to_csv(out_path, index=False)

    # Imprime confirmacao para o usuario
    print(f"Dataset salvo em: {out_path}")
    print(f"Shape: {df.shape}")  # (linhas, colunas)
    print(df.head())              # mostra as 5 primeiras linhas como preview


# Bloco especial do Python: so executa se este arquivo for chamado diretamente
# (nao executa quando o arquivo e importado por outro modulo)
if __name__ == "__main__":
    download()
