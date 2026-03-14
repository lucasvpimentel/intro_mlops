# 04 — Conceito de MLOps

## O que e MLOps?

MLOps (Machine Learning Operations) e o conjunto de praticas, ferramentas
e cultura que une **Ciencia de Dados** e **Engenharia de Software** para
levar modelos de ML a producao de forma confiavel, reproducivel e escalavel.

```
DevOps  +  ML  =  MLOps
(CI/CD)    (treino/inferencia)   (automacao do ciclo de vida do modelo)
```

---

## Por que MLOps existe? O problema do debito tecnico

Sem MLOps, o caminho tipico de um modelo e:

```
[Notebook do cientista] --> "funciona no meu computador" --> deploy manual
                        --> modelo roda 3 meses --> começa a errar
                        --> ninguem sabe por que --> retreino manual
                        --> repete
```

Esse ciclo gera **debito tecnico de ML** — codigo e processos frageis que
custam cada vez mais para manter.

### Os problemas tipicos sem MLOps

**1. Training-Serving Skew**
O modelo treina com uma formula de feature mas em producao a formula
e diferente (ex: normalizar com a media do dataset vs a media do mes atual).

```python
# Treino (notebook)
X["idade_norm"] = (X["idade"] - X["idade"].mean()) / X["idade"].std()

# Producao (codigo diferente, escrito por outro time)
X["idade_norm"] = (X["idade"] - 35) / 10   # valores hardcoded errados!
```

**2. Sem reproducibilidade**
"Rodei o treino semana passada e deu 94% de accuracy. Hoje rodei de novo
e deu 87%. Nao sei o que mudou."

Causas: dados mudaram, biblioteca atualizada, seed aleatoria nao fixada.

**3. Sem monitoramento**
O modelo vai degradando silenciosamente. O negocio so percebe quando
o impacto ja e grande (prejuizo, reclamacoes de clientes).

**4. Silos entre times**
```
Cientista de dados: "O modelo esta pronto!"
Engenheiro:         "Como eu coloco isso em producao?"
Cientista:          "Nao sei, e um notebook Jupyter..."
```

---

## O que MLOps resolve

| Problema | Solucao MLOps |
|---|---|
| Training-serving skew | Feature Store + pipeline unificado |
| Irreprodutibilidade | Versionamento de dados, codigo e modelos |
| Sem monitoramento | Data drift + model drift automatizados |
| Silos entre times | Plataforma compartilhada + CI/CD |
| Deploy manual | Pipeline automatizado de CI/CD/CT |

---

## Os tres niveis de maturidade MLOps

### Nivel 0 — Processo manual

```
Notebook --> script manual --> servidor --> pronto
(comum em projetos iniciais e provas de conceito)
```
- Treino e deploy feitos manualmente pelo cientista
- Sem versionamento, sem monitoramento, sem automacao
- Funciona para 1-2 modelos; nao escala

### Nivel 1 — Pipeline automatizado de ML

```
Dados novos --> Pipeline de treino automatico --> novo modelo --> deploy automatico
```
- Pipeline de treino e codigo (nao mais notebook)
- Treinamento Continuo (CT): modelo retreina automaticamente com dados novos
- Monitoramento basico (alertas de drift)
- Bom para times com 3-10 modelos em producao

### Nivel 2 — CI/CD de ML completo

```
Pull Request --> Testes automaticos --> Pipeline de CI --> Treino --> Avaliacao
             --> A/B test --> Deploy gradual --> Monitoramento --> Rollback auto
```
- Cada mudanca de codigo ou dado dispara um pipeline completo
- Testes automaticos de qualidade de modelo antes do deploy
- Deploy gradual (canary, blue-green)
- Para times com dezenas de modelos em producao

---

## MLOps na pratica — exemplo com os exercicios deste repositorio

Os exercicios 01-06 implementam os principios do Nivel 0/1:

```
Nivel 0 (o que fizemos):
  python main.py pipeline   # treino manual mas reproducivel
  python main.py predict    # inferencia isolada dos dados de treino

Nivel 1 (proximo passo):
  # Agendar retreino quando drift for detectado
  if drift_report["overall_status"] == "ALERT":
      subprocess.run(["python", "main.py", "train"])
      subprocess.run(["python", "main.py", "deploy"])
```

---

## Papeis em um time de MLOps

| Papel | Responsabilidade |
|---|---|
| Data Scientist | Experimentos, feature engineering, escolha do modelo |
| ML Engineer | Pipeline de treino, deploy, infraestrutura |
| Data Engineer | Ingestao, qualidade e disponibilidade dos dados |
| MLOps Engineer | CI/CD, monitoramento, plataforma de ML |

Em times menores, uma pessoa acumula varios desses papeis.

---

## Resumo

MLOps nao e uma ferramenta — e uma **cultura e conjunto de praticas**.
O objetivo e tratar modelos de ML com o mesmo rigor de engenharia que
aplicamos a software tradicional: versionamento, testes, automacao,
monitoramento e deploys confiaveis.

> "Um modelo que nao esta em producao nao agrega valor.
>  Um modelo em producao sem monitoramento e uma bomba-relogio."
