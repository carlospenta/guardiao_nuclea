"""
Guardiao Preditivo de Risco - Ponto de entrada principal
Challenge FIAP/Nuclea 2025 - Equipe DataMinds

Sobe a API FastAPI e o dashboard Streamlit de uma vez
"""

import subprocess
import sys
import time
import os
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def iniciar():
    processos = []

    try:
        # sobe a api primeiro
        print("=" * 60)
        print("GUARDIAO PREDITIVO DE RISCO")
        print("=" * 60)
        print()
        print("[*] Subindo API FastAPI na porta 8000...")
        proc_api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=BASE_DIR,
        )
        processos.append(proc_api)

        # espera um pouco pra api subir antes do streamlit
        time.sleep(2)

        print("[*] Subindo dashboard Streamlit na porta 8501...")
        proc_st = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app_streamlit.py",
             "--server.port", "8501", "--server.headless", "true"],
            cwd=BASE_DIR,
        )
        processos.append(proc_st)

        print()
        print("-" * 60)
        print("Tudo rodando!")
        print("  API:       http://localhost:8000")
        print("  API docs:  http://localhost:8000/docs")
        print("  Dashboard: http://localhost:8501")
        print("-" * 60)
        print("Ctrl+C pra parar tudo")
        print()

        # fica esperando ate alguem dar ctrl+c
        for p in processos:
            p.wait()

    except KeyboardInterrupt:
        print("\n[*] Parando tudo...")
        for p in processos:
            p.terminate()
        for p in processos:
            p.wait()
        print("[*] Finalizado.")


if __name__ == "__main__":
    iniciar()
