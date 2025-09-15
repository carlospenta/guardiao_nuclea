# Guardião Preditivo de Risco — Núclea FIDC

**Enterprise Challenge FIAP 2025 — Núclea**
Turma 1TSCOR | Equipe DataMinds

Modelo preditivo de inadimplencia para recebiveis de FIDCs. A ideia é transformar a Nuclea de registradora passiva em plataforma ativa de inteligencia de risco.

---

## Sobre o Projeto

FIDCs tem uma taxa media de inadimplencia de uns 9%, o que da mais ou menos R$ 6 bi em creditos atrasados. O Guardião usa dados internos da Nuclea (historico de boletos, scores, liquidez) junto com features derivadas pra tentar prever inadimplencia antes da cessão dos creditos.

### Objetivos
- Diminuir a inadimplencia media dos FIDCs (de ~9% pra algo perto de 6%)
- Enriquecer dados de recebiveis com inteligencia preditiva
- Gerar scores de risco pra ajudar na decisao de compra de recebiveis

---

## Arquitetura

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

## Dados

| Base | Registros | Descrição |
|------|-----------|-----------|
| `base_boletos_fiap.csv` | 7.118 | Boletos cedidos a FIDCs (IDs, datas, valores, tipo de baixa) |
| `base_auxiliar_fiap.csv` | 4.612 | Dados por CNPJ (CNAE, UF, scores, liquidez, inadimplencia) |

### Variavel alvo
Boleto é considerado **inadimplente** quando:
- Tipo de baixa = solicitação do cedente, decurso de prazo ou envio pra protesto
- Ou pagamento com atraso maior que 15 dias

Taxa de inadimplencia na base: **10,56%**

---

## Features

20 features divididas em 4 grupos:

| Grupo | Features | Origem |
|-------|----------|--------|
| Boleto | Valor nominal, prazo, tipo de espécie | `base_boletos` |
| Pagador | Índice de liquidez, média de atraso, share inadimplência, scores | `base_auxiliar` |
| Beneficiário | Índice de liquidez cedente, score materialidade | `base_auxiliar` |
| Agregadas | Qtd boletos, valor médio, prazo médio, taxa inadimplência histórica | Calculadas |

---

## Resultados (Sprint 3)

| Métrica | Random Forest | Gradient Boosting |
|---------|:------------:|:-------------------:|
| AUC-ROC (teste) | 0.9526 | **0.9580** |
| AUC-ROC (CV 5-fold) | — | 0.9549 ± 0.019 |
| Accuracy | 96% | 96% |
| Precision (Inadimplente) | 81% | 87% |
| Recall (Inadimplente) | 79% | 71% |

Gradient Boosting ficou como melhor modelo, AUC-ROC de 0.958.

### Graficos gerados
Ficam salvos na pasta `output/`:
- Tipos de baixa
- Valor nominal
- Top UFs
- Correlação
- Curva ROC
- Matriz de confusão
- Feature importance
- Score de risco

---

## Como rodar

### Requisitos
- Python 3.9+
- pandas, scikit-learn, matplotlib, seaborn

### Instalação
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Execução
```bash
# bota os CSVs na pasta data/ e roda:
python main.py
```

Resultados saem no terminal e graficos vao pra `output/`.

---

## Estrutura

```
guardiao_nuclea/
├── main.py                 # pipeline principal (EDA + modelo)
├── README.md
├── data/
│   ├── base_boletos_fiap.csv
│   └── base_auxiliar_fiap.csv
└── output/                 # graficos gerados
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

## Equipe

| Nome | RM |
|------|-----|
| Carlos Almeida | 568444 |
| Larissa Mota | 567514 |
| Fernanda Silva | 567661 |
| Eduardo Casagrande | 567323 |

---

## Roadmap

- [x] Sprint 1 — Ideação e contextualização
- [x] Sprint 2 — Arquitetura e protótipos
- [x] Sprint 3 — MVP com EDA e modelo preditivo *(atual)*
- [ ] Sprint 4 — Solução final, dashboard e video pitch

---

*FIAP — Enterprise Challenge Núclea 2025*
