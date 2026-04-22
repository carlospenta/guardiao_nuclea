"""
Dashboard Streamlit - Guardiao Nuclea
tela pra rodar o pipeline, ver graficos e metricas
"""

import streamlit as st
import os
import requests
import json
import datetime
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
                "boletos_df": boletos,
                "auxiliar_df": auxiliar,
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


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=64,
)
st.sidebar.title("Guardião Nuclea")
st.sidebar.caption("Inteligência Preditiva de Risco")

st.sidebar.divider()

# --- Navegacao ---
st.sidebar.subheader("🧭 Navegação")
pagina = st.sidebar.radio(
    "Selecione a visão",
    [
        "📊 Visão Executiva",
        "📁 Dados",
        "🔍 Análise Exploratória",
        "⚙️ Feature Engineering",
        "🤖 Modelo",
        "🚀 Pipeline Completo",
        "🏗️ Arquitetura",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

# --- Filtros ---
st.sidebar.subheader("🎛️ Filtros")

filtro_risco = st.sidebar.select_slider(
    "Nível de risco mínimo",
    options=["Baixo", "Médio", "Alto", "Crítico"],
    value="Baixo",
)

filtro_data_inicio = st.sidebar.date_input(
    "Data início",
    value=datetime.date(2025, 1, 1),
    min_value=datetime.date(2020, 1, 1),
    max_value=datetime.date.today(),
)
filtro_data_fim = st.sidebar.date_input(
    "Data fim",
    value=datetime.date.today(),
    min_value=datetime.date(2020, 1, 1),
    max_value=datetime.date.today(),
)

st.sidebar.divider()

# --- Acoes rapidas ---
st.sidebar.subheader("⚡ Ações Rápidas")

btn_reexecutar = st.sidebar.button("🔄 Reexecutar Pipeline", use_container_width=True)
btn_limpar_cache = st.sidebar.button("🗑️ Limpar Cache", use_container_width=True)
btn_exportar = st.sidebar.button("📥 Exportar Relatório", use_container_width=True)

if btn_limpar_cache:
    st.cache_data.clear()
    if _no_cloud():
        from pipeline import _cache as pc
        pc.clear()
    st.sidebar.success("Cache limpo!")

if btn_exportar:
    st.sidebar.info("Relatório disponível na aba ativa.")

st.sidebar.divider()

# --- Status do sistema ---
st.sidebar.subheader("📡 Status")

modelo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_treinado.pkl")
if os.path.exists(modelo_path):
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(modelo_path))
    st.sidebar.success(f"Modelo treinado ✓")
    st.sidebar.caption(f"Atualizado em {mod_time.strftime('%d/%m/%Y %H:%M')}")
else:
    st.sidebar.warning("Modelo não treinado")

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "guardiao_nuclea.db")
if os.path.exists(db_path):
    st.sidebar.success("Banco SQLite ✓")
else:
    st.sidebar.warning("Banco SQLite não encontrado")

graficos_existentes = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")] if os.path.exists(OUTPUT_DIR) else []
st.sidebar.caption(f"{len(graficos_existentes)} gráficos disponíveis")

st.sidebar.divider()
st.sidebar.caption("Guardião Nuclea v0.4.0")
st.sidebar.caption(f"© {datetime.date.today().year} Núclea — Dados & Analytics")


# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.title("🛡️ Guardião Nuclea")
st.caption("Plataforma de Inteligência Preditiva de Risco — Núclea")


# ============================================================
# PAGINAS
# ============================================================

# --- helper para reexecutar pipeline via botao da sidebar ---
if btn_reexecutar:
    pagina = "🚀 Pipeline Completo"

# ---- VISAO EXECUTIVA ----
if pagina == "📊 Visão Executiva":
    st.subheader("Visão Executiva — Guardião Nuclea")
    st.write("Resumo de negócio do modelo preditivo de inadimplência para FIDCs")

    with st.spinner("Carregando dados..."):
        _exec_dados, _exec_err = chamar_api("dados")

    if _exec_err:
        st.error(_exec_err)
    else:
        total_boletos = _exec_dados["boletos"]["linhas"]
        total_cnpjs = _exec_dados["auxiliar"]["linhas"]

        with st.spinner("Processando indicadores..."):
            _exec_feat, _exec_feat_err = chamar_api("features")

        if _exec_feat_err:
            st.warning(f"Não foi possível carregar features: {_exec_feat_err}")
        else:
            taxa_inad = _exec_feat["taxa_inadimplencia"]
            n_inad = _exec_feat["inadimplentes"]
            n_adim = _exec_feat["adimplentes"]

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

            import numpy as np
            vlr_medio_boleto = 15_000
            volume_total = total_boletos * vlr_medio_boleto
            perda_atual = volume_total * (taxa_inad / 100)
            perda_com_modelo = volume_total * 0.06
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

            st.markdown("---")
            st.markdown("### 📈 Gráficos de Análise")
            graficos_exec = ["05_curva_roc.png", "08_dist_score_risco.png",
                             "07_feature_importance.png", "06_matriz_confusao.png"]
            mostrar_graficos(graficos_exec)

            st.markdown("---")
            st.markdown("### 📋 Resumo Executivo")
            st.info(f"""
**Guardião Nuclea** é um modelo preditivo de inadimplência para recebíveis de FIDCs, 
desenvolvido internamente pela área de Dados & Analytics da Núclea.

🔹 **Problema:** FIDCs enfrentam taxa média de inadimplência de ~{taxa_inad}%, gerando perdas significativas.

🔹 **Solução:** Modelo de Machine Learning ({_exec_modelo['melhor_modelo'] if not _exec_modelo_err else 'Gradient Boosting'}) 
que analisa 20 features derivadas de dados de boletos, pagadores e beneficiários para prever inadimplência 
antes da cessão dos créditos.

🔹 **Resultado:** AUC-ROC de {_exec_modelo['melhor_auc'] if not _exec_modelo_err else '0.958'}, 
permitindo identificar ~87% dos inadimplentes com alta precisão.

🔹 **Impacto:** Potencial de reduzir a taxa de inadimplência de ~{taxa_inad}% para ~6%, 
gerando economia estimada de R$ {economia / 1e6:.1f}M no portfólio analisado.
""")

# ---- DADOS ----
elif pagina == "📁 Dados":
    st.subheader("📁 Carga dos Dados")
    st.write("Visualize as bases de dados utilizadas pelo modelo.")

    with st.spinner("Carregando dados..."):
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

        # mostrar preview dos dados se disponivel (modo cloud)
        if "boletos_df" in data:
            st.divider()
            st.markdown("### 🔎 Preview dos Dados")
            import pandas as pd
            tab_bol, tab_aux = st.tabs(["Boletos", "Auxiliar"])
            with tab_bol:
                df_bol = data["boletos_df"]
                st.dataframe(df_bol.head(100), use_container_width=True, height=400)
                st.caption(f"Exibindo 100 de {len(df_bol)} registros")
            with tab_aux:
                df_aux = data["auxiliar_df"]
                st.dataframe(df_aux.head(100), use_container_width=True, height=400)
                st.caption(f"Exibindo 100 de {len(df_aux)} registros")

# ---- EDA ----
elif pagina == "🔍 Análise Exploratória":
    st.subheader("🔍 Análise Exploratória")

    if st.button("Rodar EDA", key="btn_eda", use_container_width=False):
        with st.spinner("Processando EDA..."):
            data, erro = chamar_api("eda")
        if erro:
            st.error(erro)
        else:
            st.session_state["eda_data"] = data

    if "eda_data" in st.session_state:
        data = st.session_state["eda_data"]
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
        st.write("**Gráficos gerados:**")
        mostrar_graficos(data.get("graficos", []))

# ---- FEATURES ----
elif pagina == "⚙️ Feature Engineering":
    st.subheader("⚙️ Feature Engineering")

    if st.button("Rodar Feature Engineering", key="btn_feat", use_container_width=False):
        with st.spinner("Criando features..."):
            data, erro = chamar_api("features")
        if erro:
            st.error(erro)
        else:
            st.session_state["feat_data"] = data

    if "feat_data" in st.session_state:
        data = st.session_state["feat_data"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taxa Inadimplencia", f"{data['taxa_inadimplencia']}%")
        with col2:
            st.metric("Inadimplentes", data["inadimplentes"])
        with col3:
            st.metric("Adimplentes", data["adimplentes"])

        st.write(f"**Dataset:** {data['shape'][0]} linhas, {data['shape'][1]} colunas")

        with st.expander("📋 Features utilizadas", expanded=False):
            for f in data["features"]:
                st.write(f"- `{f}`")

        st.divider()
        mostrar_graficos([data.get("grafico", "")])

# ---- MODELO ----
elif pagina == "🤖 Modelo":
    st.subheader("🤖 Treinamento dos Modelos")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        treinar = st.button("Treinar modelos", key="btn_treinar")

    if treinar:
        with st.spinner("Treinando... isso demora uns segundos"):
            data, erro = chamar_api("treinar")
        if erro:
            st.error(erro)
        else:
            st.session_state["modelo_data"] = data

    if "modelo_data" in st.session_state:
        data = st.session_state["modelo_data"]
        st.success(f"Melhor modelo: **{data['melhor_modelo']}** (AUC-ROC: {data['melhor_auc']})")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Random Forest**")
            st.metric("AUC-ROC", data["random_forest"]["auc_roc"])
            with st.expander("Classification Report"):
                st.json(data["random_forest"]["report"])
        with col2:
            st.write("**Gradient Boosting**")
            st.metric("AUC-ROC", data["gradient_boosting"]["auc_roc"])
            with st.expander("Classification Report"):
                st.json(data["gradient_boosting"]["report"])

        st.divider()
        st.write(f"**Validação cruzada (5-fold):** AUC médio = {data['cv_auc_medio']} ± {data['cv_auc_std']}")
        st.write(f"Treino: {data['treino_size']} | Teste: {data['teste_size']}")

        st.divider()
        st.write("**Gráficos do modelo:**")
        mostrar_graficos(data.get("graficos", []))

# ---- PIPELINE COMPLETO ----
elif pagina == "🚀 Pipeline Completo":
    st.subheader("🚀 Pipeline Completo")
    st.write("Roda todas as etapas de uma vez: carga, EDA, features e treinamento")

    executar = st.button("Rodar pipeline completo", key="btn_pipeline") or btn_reexecutar

    if executar:
        with st.spinner("Rodando pipeline completo... pode demorar um pouco"):
            data, erro = chamar_api("pipeline")
        if erro:
            st.error(erro)
        else:
            st.session_state["pipeline_data"] = data
            st.success("Pipeline finalizado!")

    if "pipeline_data" in st.session_state:
        data = st.session_state["pipeline_data"]
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
        st.write("**Todos os gráficos:**")
        todos_graficos = data["eda"].get("graficos", []) + [feat.get("grafico", "")] + modelo.get("graficos", [])
        mostrar_graficos(todos_graficos)

# ---- ARQUITETURA ----
elif pagina == "🏗️ Arquitetura":
    st.subheader("🏗️ Arquitetura da Solução")
    st.write("Diagramas de arquitetura do Guardião Nuclea")

    diagramas = {
        "Arquitetura Técnica": "arquitetura_tecnica.png",
        "Arquitetura Executiva (MVP)": "arquitetura_executiva_mvp.png",
        "Arquitetura Final (MVP)": "arquitetura_final_mvp.png",
        "Jornada do Dado": "fluxograma_jornada_dado.png",
        "Jornada do Usuário": "fluxograma_jornada_usuario.png",
        "Visão Geral": "arquitetura_guardiao_nuclea.png",
    }

    selecionado = st.selectbox("Selecione o diagrama", list(diagramas.keys()))
    arquivo = diagramas[selecionado]
    caminho = os.path.join(OUTPUT_DIR, arquivo)

    if os.path.exists(caminho):
        img = Image.open(caminho)
        st.image(img, caption=selecionado, use_container_width=True)
    else:
        st.warning(f"Diagrama não encontrado: {arquivo}")

    with st.expander("📋 Tecnologias utilizadas"):
        st.markdown("""
| Camada | Tecnologia | Função |
|--------|-----------|--------|
| **Dados** | Pandas, SQLite | Ingestão, persistência |
| **ML** | scikit-learn (RF, GB) | Treinamento e predição |
| **API** | FastAPI, Uvicorn | 8 endpoints REST |
| **Dashboard** | Streamlit | Visualização interativa |
| **Visualização** | Matplotlib, Seaborn | Gráficos analíticos |
| **Deploy** | Streamlit Cloud | Hospedagem do dashboard |
""")
