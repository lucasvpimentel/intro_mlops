# Exercicio Final — Sistema de Analise de Pinguins (Palmer Station)

## 1. Contexto do Problema

A equipe de biologos da Estacao Palmer precisa de uma ferramenta automatizada
para processar dados de pinguins coletados em campo. O objetivo e criar um sistema
que, a partir de medidas fisicas (bico, asas, sexo), realize duas tarefas de uma so vez:

**Tarefa A (Classificacao)**: Identificar a especie do pinguim (Adelie, Chinstrap ou Gentoo).

**Tarefa B (Regressao)**: Estimar a massa corporal (peso) do animal para controle de saude.

## 2. Restricoes Tecnicas

**Algoritmo**: E proibido o uso de Redes Neurais (Deep Learning). Utilize modelos classicos
como Random Forest, Gradient Boosting ou SVM.

**Arquitetura**: O projeto deve seguir rigorosamente a estrutura de pastas abaixo para
garantir modularidade e reprodutibilidade.

## 3. Estrutura do Projeto

```
penguins_mlops/
├── data/
│   ├── raw/               # Dados originais (CSV do Palmer Penguins)
│   ├── processed/         # Dados limpos e prontos para os modelos
│   └── samples/           # Arquivos pequenos para testar a inferencia
├── docs/                  # Documentacao e dicionario de atributos
├── models/                # Onde serao salvos os modelos (.joblib) e scalers
├── notebooks/             # Analise exploratoria e rascunhos de codigo
├── reports/               # Graficos (Matriz de confusao e erros de peso)
├── scripts/               # Shell scripts para automatizar execucao
├── src/                   # Codigo-fonte principal
│   ├── __init__.py        # Transforma a pasta em pacote Python
│   ├── data_loader.py     # Ingestao e divisao Treino/Teste
│   ├── preprocessor.py    # TRATAMENTO DOS DADOS (Obrigatorio)
│   ├── trainer.py         # Treinamento dos dois modelos (Classif. e Regressao)
│   ├── evaluator.py       # Calculo de metricas (Acuracia e RMSE)
│   └── inference.py       # Script de uso diario (Predicao)
├── requirements.txt       # Bibliotecas necessarias (Pandas, Sklearn, Joblib)
└── main.py                # Orquestrador que executa o pipeline completo
```

## 4. Script de Tratamento de Dados: src/preprocessor.py

Este script deve ser capaz de lidar com valores ausentes e preparar as variaveis para
os modelos. A funcao principal `limpar_e_preparar(df, modo_treino)` deve:

1. **Selecionar as features** relevantes: `bill_length_mm`, `bill_depth_mm`,
   `flipper_length_mm`, `sex`, `island`

2. **Imputar valores ausentes**:
   - Numericos: preencher com a media (SimpleImputer strategy='mean')
   - Categoricos: preencher com o mais frequente (strategy='most_frequent')

3. **Codificar variaveis categoricas** (LabelEncoder para `sex` e `island`)

4. **Normalizar features** (StandardScaler)

5. **Salvar artefatos** (scalers, encoders) em `models/` no modo treino,
   e carrega-los no modo inferencia

## 5. Dicas para Implementacao do trainer.py

Como nao usaremos Redes Neurais para a multi-tarefa, ha duas opcoes:

**Opcao A (Simples)**: Treinar um `RandomForestClassifier` para a especie e um
`RandomForestRegressor` para o peso separadamente, salvando dois arquivos `.joblib`.

**Opcao B (Elegante)**: Usar o `MultiOutputRegressor` do Sklearn, embora ele funcione
melhor quando todas as saidas sao numeros.

## 6. Metricas de Avaliacao

| Tarefa          | Metrica Principal | Meta             |
|-----------------|------------------|-----------------|
| Classificacao   | Accuracy         | > 95%           |
| Regressao       | RMSE             | < 400 g         |
