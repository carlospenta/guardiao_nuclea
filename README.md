# Guardião Nuclea — Núclea FIDC

**Plataforma de Inteligência Preditiva de Risco para Recebíveis**
Desenvolvido internamente pela área de Dados & Analytics — Núclea

Modelo preditivo de inadimplencia para recebiveis de FIDCs. A ideia é transformar a Nuclea de registradora passiva em plataforma ativa de inteligencia de risco.

---

## Sobre o Projeto

FIDCs tem uma taxa media de inadimplencia de uns 9%, o que da mais ou menos R$ 6 bi em creditos atrasados. O Guardião Nuclea usa dados internos da Nuclea (historico de boletos, scores, liquidez) junto com features derivadas pra tentar prever inadimplencia antes da cessão dos creditos.

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
┌─────────────┐    ┌──────────────────┐                           ┌──────────────┐
│  Dashboard  │◀───│   API FastAPI    │◀──────────────────────────│  Score de    │
│  Streamlit  │    │   8 endpoints    │                           │  Risco &     │
│  (Cloud)    │    │                  │                           │  Alertas     │
└─────────────┘    └──────────────────┘                           └──────────────┘
```

### Diagramas de Arquitetura

O projeto inclui 5 diagramas visuais gerados automaticamente (`gerar_diagrama_arquitetura.py`):

| Diagrama | Arquivo | Descrição |
|----------|---------|-----------|
| Arquitetura Técnica | `output/arquitetura_tecnica.png` | 4 camadas (Dados, Processamento, ML, Interface) com componentes laterais |
| Jornada do Dado | `output/fluxograma_jornada_dado.png` | 8 etapas do dado bruto ao score de risco |
| Jornada do Usuário | `output/fluxograma_jornada_usuario.png` | 6 passos da experiência do analista |
| Arquitetura Executiva MVP | `output/arquitetura_executiva_mvp.png` | Visão de alto nível (Entrada → Inteligência → Saída) |
| Arquitetura Final MVP | `output/arquitetura_final_mvp.png` | 5 camadas completas com fontes externas, deploy e usuários |

Para regenerar os diagramas:
```bash
python gerar_diagrama_arquitetura.py
```

### Banco de Dados

O projeto usa **SQLite** como camada de persistencia:
- Na primeira execução, o banco é populado automaticamente a partir dos CSVs
- Flag `PROD` em `database.py` controla a fonte de dados:
  - `PROD = False` (default/MVP): lê direto dos CSVs
  - `PROD = True`: lê do SQLite

### Deploy no Streamlit Community Cloud

A aplicação roda no **Streamlit Community Cloud** sem precisar de servidor FastAPI separado. No Cloud, as funções do pipeline são chamadas diretamente (sem HTTP), mantendo o mesmo comportamento da versão local.

🔗 **App online:** [guardiao-nuclea.streamlit.app](https://guardiao-nuclea.streamlit.app/)

---

## Dashboard

O dashboard Streamlit possui 6 abas:

| Aba | Descrição |
|-----|-----------|
| 📊 **Visão Executiva** | KPIs de negócio, impacto financeiro estimado, gráficos de portfólio (pizza + barras), performance dos modelos, resumo executivo |
| **Dados** | Informações dos datasets (shape, colunas, tipos) |
| **EDA** | Análise exploratória com gráficos de tipos de baixa, valor nominal e UFs |
| **Features** | Feature engineering — 20 features, taxa de inadimplência, heatmap de correlação |
| **Modelo** | Métricas dos modelos (AUC-ROC, accuracy, precision, recall), curva ROC, matriz de confusão |
| **Pipeline Completo** | Executa todo o pipeline de uma vez |

A aba **Visão Executiva** é voltada para stakeholders e gestores, apresentando:
- KPIs: total de boletos, CNPJs analisados, taxa de inadimplência, AUC-ROC
- Impacto financeiro: volume analisado, perda estimada, economia potencial com o modelo
- Gráficos: composição do portfólio, impacto financeiro, comparação de modelos
- Resumo executivo textual

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

## Resultados

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
- Dependencias listadas em `requirements.txt`

### Instalação
```bash
pip install -r requirements.txt
```

### Execução (aplicação completa — local)
```bash
python main.py
```

Isso sobe tudo de uma vez:
- API FastAPI em http://localhost:8000 (docs em http://localhost:8000/docs)
- Dashboard Streamlit em http://localhost:8501

Ctrl+C pra parar.

### Streamlit Community Cloud
O deploy é feito apontando o Cloud para `main.py`. A aplicação detecta automaticamente o ambiente Cloud e roda o dashboard sem subir a API separada.

### Versão console (sem servidor)
```bash
python versao_console.py
```

Roda o pipeline inteiro no terminal e salva graficos em `output/`.

### Gerar diagramas de arquitetura
```bash
python gerar_diagrama_arquitetura.py
```

Gera 5 diagramas visuais na pasta `output/`.

### Endpoints da API (local)
| Rota | O que faz |
|------|----------|
| `GET /` | Health check |
| `GET /dados` | Info dos datasets |
| `GET /eda` | Roda analise exploratoria |
| `GET /features` | Feature engineering |
| `GET /treinar` | Retorna metricas do modelo salvo (ou treina se nao existir). `?forcar=true` pra retreinar |
| `GET /pipeline` | Roda tudo de uma vez |
| `GET /graficos` | Lista graficos gerados |
| `GET /grafico/{nome}` | Retorna um grafico PNG |

---

## Estrutura

```
guardiao_nuclea/
├── main.py                         # ponto de entrada (sobe API + Streamlit local / Cloud)
├── versao_console.py               # pipeline completo no terminal
├── pipeline.py                     # funcoes do pipeline (reusavel)
├── api.py                          # API FastAPI
├── app_streamlit.py                # dashboard Streamlit (6 abas incl. Visão Executiva)
├── database.py                     # modulo SQLite (carga e persistencia)
├── gerar_diagrama_arquitetura.py   # gera 5 diagramas de arquitetura em PNG
├── modelo_treinado.pkl             # modelo salvo (carrega automatico)
├── requirements.txt                # dependencias Python
├── README.md
├── data/
│   ├── base_boletos_fiap.csv
│   ├── base_auxiliar_fiap.csv
│   └── guardiao_nuclea.db          # banco SQLite (gerado automaticamente)
└── output/                         # graficos e diagramas gerados
    ├── 01_tipos_baixa.png
    ├── 02_dist_valor_nominal.png
    ├── 03_top_ufs.png
    ├── 04_correlacao.png
    ├── 05_curva_roc.png
    ├── 06_matriz_confusao.png
    ├── 07_feature_importance.png
    ├── 08_dist_score_risco.png
    ├── arquitetura_tecnica.png
    ├── fluxograma_jornada_dado.png
    ├── fluxograma_jornada_usuario.png
    ├── arquitetura_executiva_mvp.png
    ├── arquitetura_final_mvp.png
    └── arquitetura_guardiao_nuclea.png
```

---

## Gestão do Projeto

O projeto utiliza **GitHub Projects** com board Kanban para gestão ágil, com issues organizadas por milestones, labels e datas.

📋 **Board:** [GitHub Projects — Guardião Nuclea](https://github.com/carlospenta/guardiao_nuclea/projects)

### Labels
| Label | Descrição |
|-------|-----------|
| `Docs` | Documentação |
| `ML` | Machine Learning |
| `Backend` | Backend/API |
| `Frontend` | Frontend/Dashboard |
| `Infra` | Infraestrutura/Deploy |
| `Arquitetura` | Arquitetura da solução |
| `Gestão` | Gestão de projeto |
| `QA` | Testes e qualidade |
| `Data` | Dados |

---

## Equipe

| Nome | Papel |
|------|-------|
| Carlos Almeida | Data Engineer / ML |
| Larissa Mota | Frontend / Docs |
| Eduardo Casagrande | Backend / QA |

---

## Roadmap

- [x] Fase 1 — Ideação e contextualização
- [x] Fase 2 — Arquitetura e protótipos
- [x] Fase 3 — MVP com EDA, modelo preditivo, API FastAPI, dashboard Streamlit (incl. Visão Executiva), SQLite, diagramas de arquitetura e deploy Cloud *(atual)*
- [ ] Fase 4 — Solução final e apresentação

---

*Núclea — Inteligência Preditiva de Risco para FIDCs*
