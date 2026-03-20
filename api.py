"""
API FastAPI do Guardiao Nuclea
endpoints pra rodar cada etapa do pipeline
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from pipeline import (
    carregar_dados,
    eda,
    feature_engineering,
    treinar_modelos,
    rodar_pipeline_completo,
    OUTPUT_DIR,
    _cache,
)

app = FastAPI(
    title="Guardiao Nuclea - API",
    description="API do modelo de inadimplencia pra FIDCs - Challenge FIAP/Nuclea 2026",
    version="0.3.0",
)

# servir os graficos como arquivos estaticos
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/")
def root():
    return {"msg": "Guardiao Nuclea - API rodando", "version": "0.3.0"}


@app.get("/dados")
def get_dados():
    """carrega os dados e retorna info basica"""
    boletos, auxiliar = carregar_dados()
    return {
        "boletos": {"linhas": boletos.shape[0], "colunas": boletos.shape[1]},
        "auxiliar": {"linhas": auxiliar.shape[0], "colunas": auxiliar.shape[1]},
        "colunas_boletos": list(boletos.columns),
        "colunas_auxiliar": list(auxiliar.columns),
    }


@app.get("/eda")
def get_eda():
    """roda analise exploratoria e retorna estatisticas"""
    boletos, auxiliar = carregar_dados()
    resultado = eda(boletos, auxiliar)
    # tirar o describe pq eh muito grande pro json
    resultado.pop("boletos_describe", None)
    return resultado


@app.get("/features")
def get_features():
    """roda feature engineering e retorna info"""
    boletos, auxiliar = carregar_dados()
    _, info = feature_engineering(boletos, auxiliar)
    return info


@app.get("/treinar")
def get_treinar(forcar: bool = False):
    """treina os modelos e retorna metricas. se ja tem modelo salvo, retorna as metricas sem retreinar (passa ?forcar=true pra forcar)"""
    if not forcar and "metricas_salvas" in _cache:
        return _cache["metricas_salvas"]
    boletos, auxiliar = carregar_dados()
    feature_engineering(boletos, auxiliar)
    resultado = treinar_modelos()
    return resultado


@app.get("/pipeline")
def get_pipeline():
    """roda o pipeline completo de uma vez"""
    resultado = rodar_pipeline_completo()
    # tirar describe pra nao estourar o json
    resultado["eda"].pop("boletos_describe", None)
    return resultado


@app.get("/graficos")
def listar_graficos():
    """lista os graficos disponiveis na pasta output"""
    arquivos = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    arquivos.sort()
    return {"graficos": arquivos, "base_url": "/output/"}


@app.get("/grafico/{nome}")
def get_grafico(nome: str):
    """retorna um grafico especifico"""
    caminho = os.path.join(OUTPUT_DIR, nome)
    if not os.path.exists(caminho):
        return JSONResponse(status_code=404, content={"erro": f"grafico {nome} nao encontrado"})
    return FileResponse(caminho, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
