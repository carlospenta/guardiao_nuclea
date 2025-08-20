# 🛡️ Guardião Preditivo de Risco — Núclea FIDC

**Enterprise Challenge FIAP 2025 — Núclea**
Turma 1TSCOR | Equipe **DataMinds**

> Modelo preditivo de inadimplência para recebíveis de FIDCs, transformando a Núclea de registradora passiva em plataforma ativa de inteligência de risco.

---

## 📋 Sobre o Projeto

Os Fundos de Investimento em Direitos Creditórios (FIDCs) enfrentam uma taxa média de inadimplência de **9,04%**, totalizando aproximadamente R$ 6 bilhões em créditos atrasados. O **Guardião Preditivo de Risco** é um serviço de alerta preditivo que combina dados internos da Núclea (histórico de boletos, scores, liquidez) com features derivadas para antecipar o risco de inadimplência antes da cessão dos créditos.

### Objetivos
- Reduzir a inadimplência média dos FIDCs de ~9% para patamares próximos de 6%
- Enriquecer dados de recebíveis com inteligência preditiva
- Fornecer scores de risco para apoiar decisões de compra de recebíveis

---

## 🏗️ Arquitetura da Solução

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Fontes de  │───▶│   Ingestão &     │───▶│  Feature Eng.   │───▶│   Modelo     │
│   Dados     │    │   Limpeza        │    │  & Enrichment   │    │  Preditivo   │
│             │    │                  │    │                 │    │              │
│ • PCR       │    │ • Pandas         │    │ • 20 features   │    │ • Random     │
│ • Base Aux  │    │ • Tratamento     │    │ • Agregações    │    │   Forest     │
│ • Scores    │    │   de nulos       │    │ • Cruzamento    │    │ • Gradient   │
│   Núclea    │    │ • Parse datas    │    │   pagador/benef │    │   Boosting   │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────┬───────┘
                                                                         │
                                                                         ▼
                                                                  ┌──────────────┐
                                                                  │  Score de    │
                                                                  │  Risco &     │
                                                                  │  Alertas     │
                                                                  └──────────────┘
```

---

## 📊 Dados Utilizados

| Base | Registros | Descrição |
|------|-----------|-----------|
| `base_boletos_fiap.csv` | 7.118 | Boletos cedidos a FIDCs (IDs, datas, valores, tipo de baixa) |
| `base_auxiliar_fiap.csv` | 4.612 | Dados complementares por CNPJ (CNAE, UF, scores, liquidez, inadimplência) |

### Variável-Alvo
Um boleto é classificado como **inadimplente** quando:
- O tipo de baixa indica solicitação do cedente, decurso de prazo ou envio para protesto; **ou**
- O pagamento ocorreu com atraso superior a 15 dias após o vencimento.

**Taxa de inadimplência observada na base: 10,56%**

---

## 🔬 Features do Modelo

O modelo utiliza **20 features** organizadas em 4 grupos:

| Grupo | Features | Origem |
|-------|----------|--------|
| **Boleto** | Valor nominal, prazo, tipo de espécie | `base_boletos` |
| **Pagador (Sacado)** | Índice de liquidez, média de atraso, share inadimplência, scores | `base_auxiliar` |
| **Beneficiário (Cedente)** | Índice de liquidez cedente, score materialidade | `base_auxiliar` |
| **Agregadas** | Qtd boletos, valor médio, prazo médio, taxa inadimplência histórica | Calculadas |

---

## 🤖 Resultados do Modelo (MVP Sprint 3)

| Métrica | Random Forest | Gradient Boosting ★ |
|---------|:------------:|:-------------------:|
| **AUC-ROC (teste)** | 0.9526 | **0.9580** |
| **AUC-ROC (CV 5-fold)** | — | **0.9549 ± 0.019** |
| **Accuracy** | 96% | 96% |
| **Precision (Inadimplente)** | 81% | 87% |
| **Recall (Inadimplente)** | 79% | 71% |

> O modelo Gradient Boosting foi selecionado como melhor modelo com AUC-ROC de **0.958**.

### Visualizações Geradas
Os gráficos são salvos automaticamente na pasta `output/`:
- Distribuição dos tipos de baixa
- Distribuição do valor nominal
- Top UFs da base auxiliar
- Matriz de correlação
- Curva ROC comparativa
- Matriz de confusão
- Importância das features
- Distribuição do score de risco

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.9+
- Bibliotecas: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Instalação
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Execução
```bash
# Coloque os CSVs na pasta data/ e execute:
python main.py
```

Os resultados serão exibidos no terminal e os gráficos salvos em `output/`.

---

## 📁 Estrutura do Projeto

```
guardiao_nuclea/
├── main.py                 # Pipeline completo (EDA + modelo preditivo)
├── README.md               # Documentação do projeto
├── data/
│   ├── base_boletos_fiap.csv
│   └── base_auxiliar_fiap.csv
└── output/                 # Gráficos gerados automaticamente
    ├── 01_tipos_baixa.png
    ├── 02_dist_valor_nominal.png
    ├── 03_top_ufs.png
    ├── 04_correlacao.png
    ├── 05_curva_roc.png
    ├── 06_matriz_confusao.png
    ├── 07_feature_importance.png
    └── 08_dist_score_risco.png
```

---

## 👥 Equipe DataMinds

| Nome | RM |
|------|-----|
| Carlos Almeida | 568444 |
| Larissa Mota | 567514 |
| Fernanda Silva | 567661 |
| Eduardo Casagrande | 567323 |

---

## 📅 Roadmap

- [x] **Sprint 1** — Ideação e contextualização do problema
- [x] **Sprint 2** — Arquitetura da solução e protótipos
- [x] **Sprint 3** — MVP preliminar com EDA e modelo preditivo *(atual)*
- [ ] **Sprint 4** — Solução final, dashboard e vídeo pitch

---

*FIAP — Enterprise Challenge Núclea 2025*
