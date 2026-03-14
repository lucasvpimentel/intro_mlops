# Dicionario de Atributos — Palmer Penguins Dataset

## Fonte

Palmer Station Antarctica LTER e K. Gorman (2020).
Dataset disponivel via seaborn: `seaborn.load_dataset("penguins")`.

## Descricao das Colunas

| Coluna               | Tipo     | Unidade | Descricao                                         |
|----------------------|----------|---------|---------------------------------------------------|
| `species`            | Categorica | —     | Especie do pinguim: Adelie, Chinstrap ou Gentoo   |
| `island`             | Categorica | —     | Ilha de coleta: Biscoe, Dream ou Torgersen        |
| `bill_length_mm`     | Numerica | mm      | Comprimento do bico (culmen)                      |
| `bill_depth_mm`      | Numerica | mm      | Profundidade (altura) do bico                     |
| `flipper_length_mm`  | Numerica | mm      | Comprimento da asa (nadadeira)                    |
| `body_mass_g`        | Numerica | gramas  | Massa corporal (peso do animal)                   |
| `sex`                | Categorica | —     | Sexo: male (macho) ou female (femea)              |
| `year`               | Numerica | ano     | Ano de coleta (2007, 2008 ou 2009) — nao usado    |

## Features Usadas pelo Modelo

O modelo usa 5 features como entrada:
- `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` (numericas → padronizadas)
- `sex`, `island` (categoricas → LabelEncoder)

## Variaveis Alvo

- **Tarefa A (Classificacao)**: `species` — identifica a especie
- **Tarefa B (Regressao)**: `body_mass_g` — estima o peso corporal

## Estatisticas Descritivas (Dataset Completo — 344 registros, 11 com NaN)

| Feature              | Media   | Std     | Min    | Max    |
|----------------------|---------|---------|--------|--------|
| bill_length_mm       | 43.9 mm | 5.5 mm  | 32.1   | 59.6   |
| bill_depth_mm        | 17.2 mm | 1.97 mm | 13.1   | 21.5   |
| flipper_length_mm    | 201 mm  | 14 mm   | 172    | 231    |
| body_mass_g          | 4202 g  | 802 g   | 2700   | 6300   |

## Distribuicao por Especie

| Especie   | Contagem | % do total |
|-----------|----------|-----------|
| Adelie    | 152      | 44%       |
| Gentoo    | 124      | 36%       |
| Chinstrap | 68       | 20%       |

## Valores Ausentes

11 registros tem pelo menos um valor nulo (principalmente em `sex`).
A estrategia de tratamento esta em `src/preprocessor.py`:
- Numericos: substituidos pela media (SimpleImputer strategy="mean")
- Categoricos: substituidos pelo mais frequente (strategy="most_frequent")
