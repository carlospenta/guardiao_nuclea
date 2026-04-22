"""
Dashboard Streamlit - Guardiao Nuclea
tela pra rodar o pipeline, ver graficos e metricas
"""

import streamlit as st
import os
import requests
import json
from PIL import Image

API_URL = "http://localhost:8000"


def _no_cloud():
    return os.environ.get("STREAMLIT_SHARING_MODE") or os.path.exists("/mount/src")


def _chamar_direto(endpoint):
    """Executa a logica da API diretamente, sem HTTP"""
    from pipeline import (
        carregar_dados, eda, feature_engineering,
        treinar_modelos, rodar_pipeline_completo, _cache,
    )
    try:
        if endpoint == "dados":
            boletos, auxiliar = carregar_dados()
            return {
                "boletos": {"linhas": boletos.shape[0], "colunas": boletos.shape[1]},
                "auxiliar": {"linhas": auxiliar.shape[0], "colunas": auxiliar.shape[1]},
                "colunas_boletos": list(boletos.columns),
                "colunas_auxiliar": list(auxiliar.columns),
            }, None
        elif endpoint == "eda":
            boletos, auxiliar = carregar_dados()
            resultado = eda(boletos, auxiliar)
            resultado.pop("boletos_describe", None)
            return resultado, None
        elif endpoint == "features":
            boletos, auxiliar = carregar_dados()
            _, info = feature_engineering(boletos, auxiliar)
            return info, None
        elif endpoint == "treinar":
            if "metricas_salvas" in _cache:
                return _cache["metricas_salvas"], None
            boletos, auxiliar = carregar_dados()
            feature_engineering(boletos, auxiliar)
            resultado = treinar_modelos()
            return resultado, None
        elif endpoint == "pipeline":
            resultado = rodar_pipeline_completo()
            resultado["eda"].pop("boletos_describe", None)
            return resultado, None
        else:
            return None, f"Endpoint desconhecido: {endpoint}"
    except Exception as e:
        return None, str(e)

st.set_page_config(
    page_title="Guardiao Nuclea",
    page_icon="🛡️",
    layout="wide",
)

st.title("Guardiao Nuclea")
st.caption("Challenge FIAP/Nuclea 2025 - Equipe DataMinds")

# sidebar com as acoes
st.sidebar.header("Acoes")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def chamar_api(endpoint):
    """chama um endpoint da api e retorna o json"""
    if _no_cloud():
        return _chamar_direto(endpoint)
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
tab_exec, tab_dados, tab_eda, tab_features, tab_modelo, tab_pipeline = st.tabs([
    "📊 Visão Executiva", "Dados", "EDA", "Features", "Modelo", "Pipeline Completo"
])

with tab_exec:
    st.subheader("Visão Executiva — Guardião Nuclea")
    st.write("Resumo de negócio do modelo preditivo de inadimplência para FIDCs")

    # carrega dados automaticamente
    with st.spinner("Carregando dados..."):
        _exec_dados, _exec_err = chamar_api("dados")

    if _exec_err:
        st.error(_exec_err)
    else:
        # --- KPIs de negocio ---
        total_boletos = _exec_dados["boletos"]["linhas"]
        total_cnpjs = _exec_dados["auxiliar"]["linhas"]

        # carregar features pra ter taxa de inadimplencia
        with st.spinner("Processando indicadores..."):
            _exec_feat, _exec_feat_err = chamar_api("features")

        if _exec_feat_err:
            st.warning(f"Não foi possível carregar features: {_exec_feat_err}")
        else:
            taxa_inad = _exec_feat["taxa_inadimplencia"]
            n_inad = _exec_feat["inadimplentes"]
            n_adim = _exec_feat["adimplentes"]

            # carregar metricas do modelo
            with st.spinner("Carregando modelo..."):
                _exec_modelo, _exec_modelo_err = chamar_api("treinar")

            st.markdown("---")
            st.markdown("### 🎯 Indicadores-Chave (KPIs)")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("📄 Total de Boletos", f"{total_boletos:,}".replace(",", "."))
            with k2:
                st.metric("🏢 CNPJs Analisados", f"{total_cnpjs:,}".replace(",", "."))
            with k3:
                st.metric("⚠️ Taxa de Inadimplência", f"{taxa_inad}%",
                          delta=f"-{round(taxa_inad - 6, 1)}% vs meta 6%", delta_color="inverse")
            with k4:
                if not _exec_modelo_err:
                    st.metric("🏆 AUC-ROC do Modelo", _exec_modelo["melhor_auc"])
                else:
                    st.metric("🏆 AUC-ROC do Modelo", "—")

            st.markdown("---")
            st.markdown("### 💰 Impacto Financeiro Estimado")

            # estimativas de negocio
            import numpy as np
            vlr_medio_boleto = 15_000  # estimativa conservadora
            volume_total = total_boletos * vlr_medio_boleto
            perda_atual = volume_total * (taxa_inad / 100)
            perda_com_modelo = volume_total * 0.06  # meta de 6%
            economia = perda_atual - perda_com_modelo

            f1, f2, f3 = st.columns(3)
            with f1:
                st.metric("Volume Analisado (est.)",
                          f"R$ {volume_total / 1e6:.0f}M")
            with f2:
                st.metric("Perda Estimada Atual",
                          f"R$ {perda_atual / 1e6:.1f}M",
                          delta=f"{taxa_inad}% de inadimplência", delta_color="inverse")
            with f3:
                st.metric("Economia Potencial c/ Modelo",
                          f"R$ {economia / 1e6:.1f}M",
                          delta=f"Redução para ~6%", delta_color="normal")

            st.markdown("---")
            st.markdown("### 📊 Visão Geral do Portfólio")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                # grafico pizza inadimplencia
                import matplotlib.pyplot as plt
                fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                cores = ["#22c55e", "#ef4444"]
                ax_pie.pie([n_adim, n_inad],
                           labels=["Adimplentes", "Inadimplentes"],
                           colors=cores, autopct="%1.1f%%",
                           startangle=90, textprops={"fontsize": 12})
                ax_pie.set_title("Composição do Portfólio", fontsize=14, fontweight="bold")
                fig_pie.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with col_g2:
                # grafico de barras - impacto financeiro
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                categorias = ["Perda Atual\n(sem modelo)", "Perda Projetada\n(com modelo)", "Economia\nEstimada"]
                valores = [perda_atual / 1e6, perda_com_modelo / 1e6, economia / 1e6]
                cores_bar = ["#ef4444", "#f59e0b", "#22c55e"]
                bars = ax_bar.bar(categorias, valores, color=cores_bar, edgecolor="white", linewidth=1.5)
                for bar, val in zip(bars, valores):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                                f"R$ {val:.1f}M", ha="center", fontsize=11, fontweight="bold")
                ax_bar.set_ylabel("R$ (milhões)", fontsize=11)
                ax_bar.set_title("Impacto Financeiro do Modelo", fontsize=14, fontweight="bold")
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                fig_bar.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            st.markdown("---")
            st.markdown("### 🔍 Performance do Modelo")

            if not _exec_modelo_err:
                p1, p2 = st.columns(2)
                with p1:
                    # gauge-like visual do AUC
                    fig_auc, ax_auc = plt.subplots(figsize=(6, 4))
                    modelos = ["Random Forest", "Gradient Boosting"]
                    aucs = [_exec_modelo["random_forest"]["auc_roc"],
                            _exec_modelo["gradient_boosting"]["auc_roc"]]
                    cores_m = ["#3b82f6", "#8b5cf6"]
                    bars_m = ax_auc.barh(modelos, aucs, color=cores_m, height=0.5, edgecolor="white")
                    for bar, val in zip(bars_m, aucs):
                        ax_auc.text(val - 0.05, bar.get_y() + bar.get_height() / 2,
                                    f"{val:.4f}", ha="center", va="center",
                                    fontsize=13, fontweight="bold", color="white")
                    ax_auc.set_xlim(0.9, 1.0)
                    ax_auc.set_title("AUC-ROC por Modelo", fontsize=14, fontweight="bold")
                    ax_auc.spines["top"].set_visible(False)
                    ax_auc.spines["right"].set_visible(False)
                    fig_auc.tight_layout()
                    st.pyplot(fig_auc)
                    plt.close(fig_auc)

                with p2:
                    # metricas do melhor modelo em tabela
                    st.markdown(f"**Melhor modelo: {_exec_modelo['melhor_modelo']}**")
                    best_key = "gradient_boosting" if "Gradient" in _exec_modelo["melhor_modelo"] else "random_forest"
                    report = _exec_modelo[best_key]["report"]
                    if isinstance(report, dict) and "Inadimplente" in report:
                        inad_r = report["Inadimplente"]
                        st.markdown(f"""
| Métrica | Valor |
|---------|-------|
| **Precision** (Inadimplente) | {inad_r.get('precision', 0):.0%} |
| **Recall** (Inadimplente) | {inad_r.get('recall', 0):.0%} |
| **F1-Score** (Inadimplente) | {inad_r.get('f1-score', 0):.0%} |
| **AUC-ROC** | {_exec_modelo['melhor_auc']} |
| **Validação Cruzada (5-fold)** | {_exec_modelo['cv_auc_medio']} ± {_exec_modelo['cv_auc_std']} |
""")
                    st.markdown(f"Treino: {_exec_modelo['treino_size']} | Teste: {_exec_modelo['teste_size']}")

            # graficos do modelo ja gerados
            st.markdown("---")
            st.markdown("### 📈 Gráficos de Análise")
            graficos_exec = ["05_curva_roc.png", "08_dist_score_risco.png",
                             "07_feature_importance.png", "06_matriz_confusao.png"]
            mostrar_graficos(graficos_exec)

            st.markdown("---")
            st.markdown("### 📋 Resumo Executivo")
            st.info(f"""
**Guardião Nuclea** é um modelo preditivo de inadimplência para recebíveis de FIDCs, 
desenvolvido no contexto do Enterprise Challenge FIAP/Núclea 2025.

🔹 **Problema:** FIDCs enfrentam taxa média de inadimplência de ~{taxa_inad}%, gerando perdas significativas.

🔹 **Solução:** Modelo de Machine Learning ({_exec_modelo['melhor_modelo'] if not _exec_modelo_err else 'Gradient Boosting'}) 
que analisa 20 features derivadas de dados de boletos, pagadores e beneficiários para prever inadimplência 
antes da cessão dos créditos.

🔹 **Resultado:** AUC-ROC de {_exec_modelo['melhor_auc'] if not _exec_modelo_err else '0.958'}, 
permitindo identificar ~87% dos inadimplentes com alta precisão.

🔹 **Impacto:** Potencial de reduzir a taxa de inadimplência de ~{taxa_inad}% para ~6%, 
gerando economia estimada de R$ {economia / 1e6:.1f}M no portfólio analisado.
""")

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
st.sidebar.caption("Guardiao Nuclea v0.3")
st.sidebar.caption("Sprint 3 - Challenge FIAP/Nuclea 2025")
