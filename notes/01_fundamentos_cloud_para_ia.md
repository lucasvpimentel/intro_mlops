# 01 — Fundamentos de Cloud para IA

## O que e Cloud Computing?

Cloud computing e o uso de servidores remotos (hospedados em datacenters)
para armazenar, processar e analisar dados — em vez de usar maquinas locais.
Para IA, isso significa acessar GPUs poderosas sob demanda, sem comprar hardware.

---

## Os dois grandes modelos de computacao para IA

### Serverless — "So pague quando rodar"

Voce sobe uma funcao ou modelo e a nuvem cuida de tudo: escalabilidade,
alocacao de recursos, disponibilidade. Voce nao gerencia servidores.

**Como funciona:**
```
Requisicao chega --> Nuvem aloca recursos --> Executa --> Libera recursos
(voce paga apenas pelo tempo de execucao)
```

**Exemplos de servicos serverless para IA:**
- AWS Lambda + SageMaker Serverless Inference
- Google Cloud Functions + Vertex AI Endpoints (modo serverless)
- Azure Functions + Azure ML Managed Endpoints

**Quando usar:**
- Inferencia com demanda variavel (picos e vales ao longo do dia)
- Modelos leves que respondem em milissegundos
- Prototipos e MVPs onde o custo precisa ser minimo

**Limitacoes:**
- Cold start: a primeira requisicao apos inatividade e mais lenta
- Nao serve para modelos muito grandes (limite de memoria/tempo)
- GPU geralmente nao disponivel no modo puramente serverless

---

### Instancias com GPU Dedicada — "Servidor sempre ligado"

Voce aluga uma maquina virtual com GPU fisica reservada para voce.
O servidor fica ativo o tempo todo — voce paga por hora, use ou nao.

**Tipos de GPU mais comuns na nuvem:**

| Instancia | GPU | Uso tipico |
|---|---|---|
| AWS `p3.2xlarge` | NVIDIA V100 | Treino de modelos medios |
| AWS `p4d.24xlarge` | 8x A100 | Treino de LLMs |
| GCP `a2-highgpu-1g` | A100 40GB | Treino e inferencia pesada |
| Azure `NC6s_v3` | V100 | Treino geral |

**Quando usar:**
- Treino de modelos grandes (horas ou dias de GPU)
- Inferencia com baixa latencia e alto volume continuo
- Fine-tuning de LLMs e modelos de visao computacional

**Exemplo pratico:**
```bash
# Treinar um modelo na AWS com GPU dedicada
# 1. Subir instancia p3.2xlarge (1x V100, 16GB VRAM)
# 2. Conectar via SSH
# 3. Rodar o treino

python train.py --epochs 100 --batch-size 256
# GPU processa ~5000 amostras/segundo vs ~200 na CPU
```

---

## Comparativo direto

| Criterio | Serverless | GPU Dedicada |
|---|---|---|
| Custo base | Zero (paga por uso) | Alto (por hora, sempre) |
| Escalabilidade | Automatica | Manual |
| Latencia | Alta no cold start | Baixa e previsivel |
| Tamanho do modelo | Limitado (~1GB) | Sem limite pratico |
| Ideal para | Inferencia esporadica | Treino e inferencia continua |

---

## Arquitetura hibrida (mais comum na pratica)

Na pratica, a maioria dos sistemas usa os dois modelos juntos:

```
[Treino]          --> GPU dedicada (p4d, A100) por horas/dias
[Inferencia leve] --> Serverless (Lambda, Cloud Functions)
[Inferencia pesada] --> GPU dedicada sempre ligada
[Armazenamento]   --> S3, GCS, Azure Blob (storage barato)
```

**Exemplo real:** um banco treina seu modelo de fraude toda semana
em uma instancia p3 (GPU dedicada, ~$3/hora por 6 horas = $18).
Depois serve o modelo via endpoint serverless que custa fracao de centavo
por chamada — e escala automaticamente nas horas de pico.

---

## Conceitos basicos de nuvem que todo ML Engineer precisa saber

| Conceito | O que e | Exemplo |
|---|---|---|
| **Regiao** | Localizacao fisica do datacenter | us-east-1 (Virginia), sa-east-1 (Sao Paulo) |
| **Zona de disponibilidade** | Datacenter isolado dentro de uma regiao | us-east-1a, us-east-1b |
| **VPC** | Rede privada virtual isolada | Seus servidores se falam sem expor a internet |
| **IAM** | Controle de quem acessa o que | "Este servico pode ler S3 mas nao pode deletar" |
| **Object Storage** | Armazenamento de arquivos ilimitado | S3, GCS, Azure Blob |
| **Container** | Pacote com codigo + dependencias | Docker com seu modelo |
| **Orquestrador** | Gerencia containers em escala | Kubernetes, ECS |
