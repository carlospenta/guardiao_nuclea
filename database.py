"""
Modulo de banco de dados SQLite - Guardiao Nuclea
Popula o SQLite a partir dos CSVs na primeira execucao.
Quando PROD=True, usa SQLite como fonte de dados.
Quando PROD=False (default/MVP), faz fallback pro CSV.
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "guardiao_nuclea.db")

BOLETOS_CSV = os.path.join(DATA_DIR, "base_boletos_fiap.csv")
AUXILIAR_CSV = os.path.join(DATA_DIR, "base_auxiliar_fiap.csv")

# Flag de producao: False = fallback CSV (MVP), True = usa SQLite
PROD = False


def get_connection():
    """Retorna conexao com o SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def banco_vazio(conn):
    """Verifica se as tabelas existem e tem dados"""
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('boletos', 'auxiliar')"
        )
        tabelas = [row[0] for row in cur.fetchall()]
        if len(tabelas) < 2:
            return True
        for tabela in tabelas:
            cur = conn.execute(f"SELECT COUNT(*) FROM {tabela}")
            if cur.fetchone()[0] == 0:
                return True
        return False
    except Exception:
        return True


def popular_banco():
    """Popula o SQLite a partir dos CSVs se o banco estiver vazio"""
    conn = get_connection()
    try:
        if not banco_vazio(conn):
            print("[*] SQLite ja populado, pulando carga dos CSVs")
            conn.close()
            return False

        print("[*] Banco vazio, populando SQLite a partir dos CSVs...")

        boletos = pd.read_csv(BOLETOS_CSV)
        boletos.to_sql("boletos", conn, if_exists="replace", index=False)
        print(f"  -> boletos: {len(boletos)} registros inseridos")

        auxiliar = pd.read_csv(AUXILIAR_CSV)
        auxiliar.to_sql("auxiliar", conn, if_exists="replace", index=False)
        print(f"  -> auxiliar: {len(auxiliar)} registros inseridos")

        conn.commit()
        print("[*] SQLite populado com sucesso!")
        conn.close()
        return True
    except Exception as e:
        print(f"[!] Erro ao popular SQLite: {e}")
        conn.close()
        raise


def carregar_boletos_sqlite():
    """Carrega boletos do SQLite como DataFrame"""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM boletos", conn, parse_dates=["dt_emissao", "dt_vencimento", "dt_pagamento"])
    conn.close()
    return df


def carregar_auxiliar_sqlite():
    """Carrega auxiliar do SQLite como DataFrame"""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM auxiliar", conn)
    conn.close()
    return df


def carregar_dados_sqlite():
    """Carrega ambos os DataFrames do SQLite"""
    return carregar_boletos_sqlite(), carregar_auxiliar_sqlite()


def inicializar_banco():
    """Inicializa o banco: cria e popula se necessario"""
    os.makedirs(DATA_DIR, exist_ok=True)
    popular_banco()
