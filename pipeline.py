"""
Funcoes do pipeline de analise - separado pra reusar na API e no streamlit
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
import io
import base64
import json
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOLETOS_PATH = os.path.join(DATA_DIR, "base_boletos_fiap.csv")
AUXILIAR_PATH = os.path.join(DATA_DIR, "base_auxiliar_fiap.csv")
MODELO_PATH = os.path.join(BASE_DIR, "modelo_treinado.pkl")

TIPOS_INADIMPLENCIA = [
    "5 - Baixa integral por solicitacao do cedente",
    "7 - Baixa integral por decurso de prazo",
    "6 - Baixa integral por envio para protesto",
]

FEATURES = [
    "vlr_nominal",
    "prazo_boleto",
    "tipo_especie_enc",
    "uf_enc",
    "sacado_indice_liquidez_1m",
    "cedente_indice_liquidez_1m",
    "score_materialidade_evolucao",
    "media_atraso_dias",
    "indicador_liquidez_quantitativo_3m",
    "share_vl_inad_pag_bol_6_a_15d",
    "score_quantidade_v2",
    "score_materialidade_v2",
    "qtd_boletos",
    "vlr_nominal_medio",
    "prazo_medio",
    "ben_qtd_boletos",
    "ben_vlr_nominal_medio",
    "ben_taxa_inadimplencia",
    "cedente_indice_liquidez_1m_ben",
    "score_materialidade_v2_ben",
]

TARGET = "inadimplente"

# cache global pra nao reprocessar toda hora
_cache = {}


def carregar_modelo_salvo():
    """tenta carregar o modelo pkl se existir, pra nao treinar de novo"""
    if os.path.exists(MODELO_PATH):
        try:
            dados = joblib.load(MODELO_PATH)
            _cache["best_model"] = dados["model"]
            _cache["best_name"] = dados["nome"]
            _cache["metricas_salvas"] = dados["metricas"]
            print(f"[*] Modelo carregado do pkl: {dados['nome']}")
            return True
        except Exception as e:
            print(f"[!] Erro ao carregar pkl, vai precisar treinar: {e}")
            return False
    return False


# tenta carregar o modelo salvo ao importar o modulo
carregar_modelo_salvo()


def carregar_dados():
    """carrega os dados e retorna boletos e auxiliar.
    PROD=True -> SQLite, PROD=False -> CSV (fallback MVP)"""
    if "boletos" in _cache and "auxiliar" in _cache:
        return _cache["boletos"], _cache["auxiliar"]

    from database import PROD, inicializar_banco, carregar_dados_sqlite

    # sempre tenta inicializar o banco (popula se vazio)
    try:
        inicializar_banco()
    except Exception as e:
        print(f"[!] Aviso ao inicializar SQLite: {e}")

    if PROD:
        print("[*] Carregando dados do SQLite (PROD=True)")
        boletos, auxiliar = carregar_dados_sqlite()
    else:
        print("[*] Carregando dados do CSV (PROD=False, fallback MVP)")
        boletos = pd.read_csv(BOLETOS_PATH, parse_dates=["dt_emissao", "dt_vencimento", "dt_pagamento"])
        auxiliar = pd.read_csv(AUXILIAR_PATH)

    _cache["boletos"] = boletos
    _cache["auxiliar"] = auxiliar
    return boletos, auxiliar


def eda(boletos, auxiliar):
    """analise exploratoria - retorna dict com stats e gera graficos"""
    resultado = {}

    resultado["boletos_shape"] = list(boletos.shape)
    resultado["auxiliar_shape"] = list(auxiliar.shape)
    resultado["boletos_describe"] = boletos.describe(include="all").to_dict()
    resultado["boletos_nulos"] = boletos.isnull().sum().to_dict()
    resultado["auxiliar_nulos"] = auxiliar.isnull().sum().to_dict()
    resultado["tipos_baixa"] = boletos["tipo_baixa"].value_counts().to_dict()

    # grafico tipos de baixa
    fig, ax = plt.subplots(figsize=(10, 5))
    boletos["tipo_baixa"].value_counts().plot.barh(ax=ax, color=sns.color_palette("viridis", 7))
    ax.set_title("Distribuição dos Tipos de Baixa")
    ax.set_xlabel("Quantidade")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_tipos_baixa.png"), dpi=150)
    plt.close(fig)

    # dist valor nominal
    fig, ax = plt.subplots(figsize=(10, 5))
    boletos["vlr_nominal"].clip(upper=boletos["vlr_nominal"].quantile(0.99)).hist(
        bins=50, ax=ax, color="#3b82f6", edgecolor="white"
    )
    ax.set_title("Distribuição do Valor Nominal dos Boletos (até P99)")
    ax.set_xlabel("Valor (R$)")
    ax.set_ylabel("Frequência")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_dist_valor_nominal.png"), dpi=150)
    plt.close(fig)

    # top UFs
    fig, ax = plt.subplots(figsize=(10, 6))
    auxiliar["uf"].value_counts().head(15).plot.barh(ax=ax, color=sns.color_palette("rocket", 15))
    ax.set_title("Top 15 UFs — Base Auxiliar")
    ax.set_xlabel("Quantidade de CNPJs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_top_ufs.png"), dpi=150)
    plt.close(fig)

    resultado["graficos"] = ["01_tipos_baixa.png", "02_dist_valor_nominal.png", "03_top_ufs.png"]
    return resultado


def feature_engineering(boletos, auxiliar):
    """cria features e retorna o dataframe pronto pro modelo"""
    boletos = boletos.copy()
    boletos["dias_atraso"] = (boletos["dt_pagamento"] - boletos["dt_vencimento"]).dt.days
    boletos["prazo_boleto"] = (boletos["dt_vencimento"] - boletos["dt_emissao"]).dt.days

    boletos["inadimplente"] = (
        boletos["tipo_baixa"].isin(TIPOS_INADIMPLENCIA)
        | (boletos["dias_atraso"] > 15)
    ).astype(int)

    taxa_inad = boletos["inadimplente"].mean() * 100

    agg_pagador = boletos.groupby("id_pagador").agg(
        qtd_boletos=("id_boleto", "count"),
        vlr_nominal_medio=("vlr_nominal", "mean"),
        vlr_nominal_total=("vlr_nominal", "sum"),
        prazo_medio=("prazo_boleto", "mean"),
        dias_atraso_medio=("dias_atraso", "mean"),
        taxa_inadimplencia=("inadimplente", "mean"),
    ).reset_index()

    agg_beneficiario = boletos.groupby("id_beneficiario").agg(
        ben_qtd_boletos=("id_boleto", "count"),
        ben_vlr_nominal_medio=("vlr_nominal", "mean"),
        ben_taxa_inadimplencia=("inadimplente", "mean"),
    ).reset_index()

    boletos_enriq = boletos.merge(
        auxiliar, left_on="id_pagador", right_on="id_cnpj", how="left", suffixes=("", "_pag")
    )
    boletos_enriq = boletos_enriq.merge(
        auxiliar[["id_cnpj", "cedente_indice_liquidez_1m", "score_materialidade_v2"]],
        left_on="id_beneficiario", right_on="id_cnpj", how="left", suffixes=("", "_ben"),
    )
    boletos_enriq = boletos_enriq.merge(agg_pagador, on="id_pagador", how="left", suffixes=("", "_agg"))
    boletos_enriq = boletos_enriq.merge(
        agg_beneficiario, on="id_beneficiario", how="left", suffixes=("", "_agg_ben")
    )

    le_especie = LabelEncoder()
    boletos_enriq["tipo_especie_enc"] = le_especie.fit_transform(boletos_enriq["tipo_especie"].astype(str))
    le_uf = LabelEncoder()
    boletos_enriq["uf_enc"] = le_uf.fit_transform(boletos_enriq["uf"].fillna("DESCONHECIDO").astype(str))

    df_model = boletos_enriq[FEATURES + [TARGET]].copy()
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    # heatmap correlacao
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df_model.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlação — Features vs Inadimplência")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_correlacao.png"), dpi=150)
    plt.close(fig)

    info = {
        "taxa_inadimplencia": round(taxa_inad, 2),
        "inadimplentes": int(boletos["inadimplente"].sum()),
        "adimplentes": int((boletos["inadimplente"] == 0).sum()),
        "shape": list(df_model.shape),
        "features": FEATURES,
        "grafico": "04_correlacao.png",
    }

    _cache["df_model"] = df_model
    return df_model, info


def treinar_modelos(df_model=None):
    """treina RF e GB, retorna metricas e o melhor modelo"""
    if df_model is None:
        df_model = _cache.get("df_model")
    if df_model is None:
        raise ValueError("roda feature_engineering antes")

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # random forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    # gradient boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    auc_gb = roc_auc_score(y_test, y_proba_gb)

    if auc_gb >= auc_rf:
        best_model, best_name = gb, "Gradient Boosting"
        best_proba, best_pred = y_proba_gb, y_pred_gb
    else:
        best_model, best_name = rf, "Random Forest"
        best_proba, best_pred = y_proba_rf, y_pred_rf

    # graficos
    # curva ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba, auc_val in [("Random Forest", y_proba_rf, auc_rf), ("Gradient Boosting", y_proba_gb, auc_gb)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC — Modelos de Inadimplência")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_curva_roc.png"), dpi=150)
    plt.close(fig)

    # matriz confusao
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=["Adimplente", "Inadimplente"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Matriz de Confusão — {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_matriz_confusao.png"), dpi=150)
    plt.close(fig)

    # feature importance
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot.barh(ax=ax, color=sns.color_palette("viridis", len(FEATURES)))
    ax.set_title(f"Importância das Features — {best_name}")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_feature_importance.png"), dpi=150)
    plt.close(fig)

    # dist score
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(best_proba[y_test == 0], bins=50, alpha=0.6, label="Adimplente", color="#22c55e")
    ax.hist(best_proba[y_test == 1], bins=50, alpha=0.6, label="Inadimplente", color="#ef4444")
    ax.set_title("Distribuição do Score de Risco Predito")
    ax.set_xlabel("Probabilidade de Inadimplência")
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_dist_score_risco.png"), dpi=150)
    plt.close(fig)

    # cross validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    report_rf = classification_report(y_test, y_pred_rf, target_names=["Adimplente", "Inadimplente"], output_dict=True)
    report_gb = classification_report(y_test, y_pred_gb, target_names=["Adimplente", "Inadimplente"], output_dict=True)

    resultado = {
        "random_forest": {"auc_roc": round(auc_rf, 4), "report": report_rf},
        "gradient_boosting": {"auc_roc": round(auc_gb, 4), "report": report_gb},
        "melhor_modelo": best_name,
        "melhor_auc": round(max(auc_rf, auc_gb), 4),
        "cv_auc_medio": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "treino_size": int(X_train.shape[0]),
        "teste_size": int(X_test.shape[0]),
        "graficos": [
            "05_curva_roc.png", "06_matriz_confusao.png",
            "07_feature_importance.png", "08_dist_score_risco.png",
        ],
    }

    _cache["best_model"] = best_model
    _cache["best_name"] = best_name

    # salva o modelo em pkl pra nao precisar treinar toda vez
    joblib.dump({"model": best_model, "nome": best_name, "metricas": resultado}, MODELO_PATH)
    print(f"[*] Modelo salvo em {MODELO_PATH}")

    return resultado


def rodar_pipeline_completo():
    """roda tudo de uma vez e retorna resumo geral"""
    boletos, auxiliar = carregar_dados()
    eda_result = eda(boletos, auxiliar)
    df_model, feat_info = feature_engineering(boletos, auxiliar)
    modelo_result = treinar_modelos(df_model)

    return {
        "eda": eda_result,
        "features": feat_info,
        "modelo": modelo_result,
    }
