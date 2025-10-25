"""
Guardiao Preditivo de Risco - MVP Sprint 3
Challenge FIAP/Nuclea 2025 - Equipe DataMinds

Script principal do modelo de inadimplencia pra FIDCs
usa dados da PCR (boletos) + base auxiliar nuclea
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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOLETOS_PATH = os.path.join(DATA_DIR, "base_boletos_fiap.csv")
AUXILIAR_PATH = os.path.join(DATA_DIR, "base_auxiliar_fiap.csv")

# fallback pro caminho original caso nao tenha copiado os csvs
if not os.path.exists(BOLETOS_PATH):
    BOLETOS_PATH = os.path.expanduser(
        "~/Downloads/Massa_Dados_Challgenge_Nuclea_v1(1)/base_boletos_fiap.csv"
    )
    AUXILIAR_PATH = os.path.expanduser(
        "~/Downloads/Massa_Dados_Challgenge_Nuclea_v1(1)/base_auxiliar_fiap.csv"
    )

# ---- CARGA DOS DADOS ----
print("=" * 70)
print("GUARDIÃO PREDITIVO DE RISCO — MVP Sprint 3")
print("=" * 70)

print("\n[1/6] Carregando dados...")
boletos = pd.read_csv(BOLETOS_PATH, parse_dates=["dt_emissao", "dt_vencimento", "dt_pagamento"])
auxiliar = pd.read_csv(AUXILIAR_PATH)

print(f"  Boletos: {boletos.shape[0]} registros, {boletos.shape[1]} colunas")
print(f"  Auxiliar: {auxiliar.shape[0]} registros, {auxiliar.shape[1]} colunas")

# ---- EDA ----
print("\n[2/6] Analise exploratoria...")

print("\n--- Estatisticas descritivas (boletos) ---")
print(boletos.describe(include="all").to_string())

print("\n--- Nulos boletos ---")
print(boletos.isnull().sum().to_string())

print("\n--- Nulos auxiliar ---")
print(auxiliar.isnull().sum().to_string())

# grafico tipos de baixa
fig, ax = plt.subplots(figsize=(10, 5))
boletos["tipo_baixa"].value_counts().plot.barh(ax=ax, color=sns.color_palette("viridis", 7))
ax.set_title("Distribuição dos Tipos de Baixa")
ax.set_xlabel("Quantidade")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_tipos_baixa.png"), dpi=150)
plt.close(fig)
print("  ok: 01_tipos_baixa.png")

# distribuicao valor nominal (cortando outliers no p99)
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
print("  ok: 02_dist_valor_nominal.png")

# top UFs
fig, ax = plt.subplots(figsize=(10, 6))
auxiliar["uf"].value_counts().head(15).plot.barh(ax=ax, color=sns.color_palette("rocket", 15))
ax.set_title("Top 15 UFs — Base Auxiliar")
ax.set_xlabel("Quantidade de CNPJs")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_top_ufs.png"), dpi=150)
plt.close(fig)
print("  ok: 03_top_ufs.png")

# ---- FEATURE ENGINEERING ----
print("\n[3/6] Feature engineering...")

# variavel alvo - inadimplencia
# boleto inadimplente = tipo_baixa de solicitacao/decurso/protesto OU atraso > 15 dias
boletos["dias_atraso"] = (boletos["dt_pagamento"] - boletos["dt_vencimento"]).dt.days
boletos["prazo_boleto"] = (boletos["dt_vencimento"] - boletos["dt_emissao"]).dt.days

TIPOS_INADIMPLENCIA = [
    "5 - Baixa integral por solicitacao do cedente",
    "7 - Baixa integral por decurso de prazo",
    "6 - Baixa integral por envio para protesto",
]

boletos["inadimplente"] = (
    boletos["tipo_baixa"].isin(TIPOS_INADIMPLENCIA)
    | (boletos["dias_atraso"] > 15)
).astype(int)

taxa_inad = boletos["inadimplente"].mean() * 100
print(f"  Taxa inadimplencia: {taxa_inad:.2f}%")
print(f"    Inadimplentes: {boletos['inadimplente'].sum()}")
print(f"    Adimplentes: {(boletos['inadimplente'] == 0).sum()}")

# agregacoes por pagador
agg_pagador = boletos.groupby("id_pagador").agg(
    qtd_boletos=("id_boleto", "count"),
    vlr_nominal_medio=("vlr_nominal", "mean"),
    vlr_nominal_total=("vlr_nominal", "sum"),
    prazo_medio=("prazo_boleto", "mean"),
    dias_atraso_medio=("dias_atraso", "mean"),
    taxa_inadimplencia=("inadimplente", "mean"),
).reset_index()

# agregacoes por beneficiario
agg_beneficiario = boletos.groupby("id_beneficiario").agg(
    ben_qtd_boletos=("id_boleto", "count"),
    ben_vlr_nominal_medio=("vlr_nominal", "mean"),
    ben_taxa_inadimplencia=("inadimplente", "mean"),
).reset_index()

# enriquecimento com base auxiliar
boletos_enriq = boletos.merge(
    auxiliar, left_on="id_pagador", right_on="id_cnpj", how="left", suffixes=("", "_pag")
)

boletos_enriq = boletos_enriq.merge(
    auxiliar[["id_cnpj", "cedente_indice_liquidez_1m", "score_materialidade_v2"]],
    left_on="id_beneficiario",
    right_on="id_cnpj",
    how="left",
    suffixes=("", "_ben"),
)

# juntando as agregacoes
boletos_enriq = boletos_enriq.merge(agg_pagador, on="id_pagador", how="left", suffixes=("", "_agg"))
boletos_enriq = boletos_enriq.merge(
    agg_beneficiario, on="id_beneficiario", how="left", suffixes=("", "_agg_ben")
)

# encoding categoricas
le_especie = LabelEncoder()
boletos_enriq["tipo_especie_enc"] = le_especie.fit_transform(boletos_enriq["tipo_especie"].astype(str))

le_uf = LabelEncoder()
boletos_enriq["uf_enc"] = le_uf.fit_transform(boletos_enriq["uf"].fillna("DESCONHECIDO").astype(str))

# features selecionadas pro modelo
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

df_model = boletos_enriq[FEATURES + [TARGET]].copy()
df_model = df_model.fillna(df_model.median(numeric_only=True))

print(f"  Dataset final: {df_model.shape[0]} linhas, {len(FEATURES)} features")

# heatmap de correlacao
fig, ax = plt.subplots(figsize=(12, 8))
corr = df_model.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
ax.set_title("Matriz de Correlação — Features vs Inadimplência")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_correlacao.png"), dpi=150)
plt.close(fig)
print("  ok: 04_correlacao.png")

# ---- MODELAGEM ----
print("\n[4/6] Treinando modelos...")

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# random forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=["Adimplente", "Inadimplente"]))
auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"  AUC-ROC: {auc_rf:.4f}")

# gradient boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

print("\n--- Gradient Boosting ---")
print(classification_report(y_test, y_pred_gb, target_names=["Adimplente", "Inadimplente"]))
auc_gb = roc_auc_score(y_test, y_proba_gb)
print(f"  AUC-ROC: {auc_gb:.4f}")

# pega o melhor
if auc_gb >= auc_rf:
    best_model, best_name, best_proba = gb, "Gradient Boosting", y_proba_gb
    best_pred = y_pred_gb
else:
    best_model, best_name, best_proba = rf, "Random Forest", y_proba_rf
    best_pred = y_pred_rf

print(f"\n  >> Melhor modelo: {best_name} (AUC={max(auc_rf, auc_gb):.4f})")

# ---- GRAFICOS DO MODELO ----
print("\n[5/6] Gerando graficos...")

# curva ROC
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, auc_val in [
    ("Random Forest", y_proba_rf, auc_rf),
    ("Gradient Boosting", y_proba_gb, auc_gb),
]:
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
print("  ok: 05_curva_roc.png")

# matriz de confusao
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["Adimplente", "Inadimplente"],
    cmap="Blues", ax=ax
)
ax.set_title(f"Matriz de Confusão — {best_name}")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "06_matriz_confusao.png"), dpi=150)
plt.close(fig)
print("  ok: 06_matriz_confusao.png")

# importancia das features
importances = pd.Series(
    best_model.feature_importances_, index=FEATURES
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
importances.plot.barh(ax=ax, color=sns.color_palette("viridis", len(FEATURES)))
ax.set_title(f"Importância das Features — {best_name}")
ax.set_xlabel("Importância")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "07_feature_importance.png"), dpi=150)
plt.close(fig)
print("  ok: 07_feature_importance.png")

# distribuicao do score
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
print("  ok: 08_dist_score_risco.png")

# ---- VALIDACAO CRUZADA ----
print("\n[6/6] Validação cruzada (5-fold)...")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"  AUC-ROC medio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---- RESUMO ----
print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)
print(f"  Boletos: {boletos.shape[0]} registros")
print(f"  Auxiliar: {auxiliar.shape[0]} CNPJs")
print(f"  Taxa inadimplencia: {taxa_inad:.2f}%")
print(f"  Features: {len(FEATURES)}")
print(f"  Melhor modelo: {best_name}")
print(f"  AUC-ROC teste: {max(auc_rf, auc_gb):.4f}")
print(f"  AUC-ROC CV: {cv_scores.mean():.4f}")
print(f"  Graficos em: {OUTPUT_DIR}")
print("=" * 70)
