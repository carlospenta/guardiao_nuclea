"""
Dashboard Streamlit - Guardiao Preditivo de Risco
tela pra rodar o pipeline, ver graficos e metricas
"""

import streamlit as st
import os
import requests
import json
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Guardiao Preditivo de Risco",
    page_icon="🛡️",
    layout="wide",
)

st.title("Guardiao Preditivo de Risco")
st.caption("Challenge FIAP/Nuclea 2025 - Equipe DataMinds")

# sidebar com as acoes
st.sidebar.header("Acoes")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def chamar_api(endpoint):
    """chama um endpoint da api e retorna o json"""
    try:
        resp = requests.get(f"{API_URL}/{endpoint}", timeout=120)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API nao esta rodando. Suba com: python api.py"
    except Exception as e:
        return None, str(e)


def mostrar_graficos(lista_graficos):
    """mostra os graficos em grid"""
    if not lista_graficos:
        st.info("Nenhum grafico gerado ainda")
        return

    cols = st.columns(2)
    for i, nome in enumerate(lista_graficos):
        caminho = os.path.join(OUTPUT_DIR, nome)
        if os.path.exists(caminho):
            with cols[i % 2]:
                img = Image.open(caminho)
                st.image(img, caption=nome, use_container_width=True)


# ---- tabs principais ----
tab_dados, tab_eda, tab_features, tab_modelo, tab_pipeline = st.tabs([
    "Dados", "EDA", "Features", "Modelo", "Pipeline Completo"
])

with tab_dados:
    st.subheader("Carga dos Dados")
    if st.button("Carregar dados", key="btn_dados"):
        with st.spinner("Carregando..."):
            data, erro = chamar_api("dados")
        if erro:
            st.error(erro)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Boletos", f"{data['boletos']['linhas']} registros")
                st.write("Colunas:", data["colunas_boletos"])
            with col2:
                st.metric("Auxiliar", f"{data['auxiliar']['linhas']} registros")
                st.write("Colunas:", data["colunas_auxiliar"])

with tab_eda:
    st.subheader("Analise Exploratoria")
    if st.button("Rodar EDA", key="btn_eda"):
        with st.spinner("Processando EDA..."):
            data, erro = chamar_api("eda")
        if erro:
            st.error(erro)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Shape boletos:**", data["boletos_shape"])
                st.write("**Shape auxiliar:**", data["auxiliar_shape"])
            with col2:
                st.write("**Tipos de baixa:**")
                st.json(data["tipos_baixa"])

            st.write("**Nulos boletos:**")
            st.json(data["boletos_nulos"])

            st.divider()
            st.write("**Graficos gerados:**")
            mostrar_graficos(data.get("graficos", []))

with tab_features:
    st.subheader("Feature Engineering")
    if st.button("Rodar Feature Engineering", key="btn_feat"):
        with st.spinner("Criando features..."):
            data, erro = chamar_api("features")
        if erro:
            st.error(erro)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Taxa Inadimplencia", f"{data['taxa_inadimplencia']}%")
            with col2:
                st.metric("Inadimplentes", data["inadimplentes"])
            with col3:
                st.metric("Adimplentes", data["adimplentes"])

            st.write(f"**Dataset:** {data['shape'][0]} linhas, {data['shape'][1]} colunas")
            st.write("**Features usadas:**")
            for f in data["features"]:
                st.write(f"- {f}")

            st.divider()
            mostrar_graficos([data.get("grafico", "")])

with tab_modelo:
    st.subheader("Treinamento dos Modelos")
    if st.button("Treinar modelos", key="btn_treinar"):
        with st.spinner("Treinando... isso demora uns segundos"):
            data, erro = chamar_api("treinar")
        if erro:
            st.error(erro)
        else:
            st.success(f"Melhor modelo: **{data['melhor_modelo']}** (AUC-ROC: {data['melhor_auc']})")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Random Forest**")
                st.metric("AUC-ROC", data["random_forest"]["auc_roc"])
                st.json(data["random_forest"]["report"])
            with col2:
                st.write("**Gradient Boosting**")
                st.metric("AUC-ROC", data["gradient_boosting"]["auc_roc"])
                st.json(data["gradient_boosting"]["report"])

            st.divider()
            st.write(f"**Validacao cruzada (5-fold):** AUC medio = {data['cv_auc_medio']} ± {data['cv_auc_std']}")
            st.write(f"Treino: {data['treino_size']} | Teste: {data['teste_size']}")

            st.divider()
            st.write("**Graficos do modelo:**")
            mostrar_graficos(data.get("graficos", []))

with tab_pipeline:
    st.subheader("Pipeline Completo")
    st.write("Roda todas as etapas de uma vez: carga, EDA, features e treinamento")

    if st.button("Rodar pipeline completo", key="btn_pipeline"):
        with st.spinner("Rodando pipeline completo... pode demorar um pouco"):
            data, erro = chamar_api("pipeline")
        if erro:
            st.error(erro)
        else:
            st.success("Pipeline finalizado!")

            # resumo
            modelo = data["modelo"]
            feat = data["features"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Taxa Inadimplencia", f"{feat['taxa_inadimplencia']}%")
            with col2:
                st.metric("Melhor Modelo", modelo["melhor_modelo"])
            with col3:
                st.metric("AUC-ROC", modelo["melhor_auc"])
            with col4:
                st.metric("AUC CV", f"{modelo['cv_auc_medio']} ± {modelo['cv_auc_std']}")

            st.divider()
            st.write("**Todos os graficos:**")
            todos_graficos = data["eda"].get("graficos", []) + [feat.get("grafico", "")] + modelo.get("graficos", [])
            mostrar_graficos(todos_graficos)

# footer
st.sidebar.divider()
st.sidebar.caption("Guardiao Preditivo de Risco v0.3")
st.sidebar.caption("Sprint 3 - Challenge FIAP/Nuclea 2025")
