"""Gera os notebooks a partir dos scripts (formato # %%) e executa cada um, salvando as saídas.

    python scripts/gerar_notebooks.py            # todos
    python scripts/gerar_notebooks.py 03 05      # só alguns

Os notebooks NÃO são editados à mão: a fonte é o script. Por padrão, tudo usa os caches gravados (minutos).
"""
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "notebooks"
FONTES = sorted(p for p in (RAIZ / "scripts").glob("[0-9][0-9]_*.py"))
HISTORIA = RAIZ / "scripts" / "historia.py"

pedidos = sys.argv[1:]
alvos = [p for p in FONTES if not pedidos or p.name[:2] in pedidos]
if HISTORIA.exists() and (not pedidos or "historia" in pedidos):
    alvos.append(HISTORIA)
SAIDA.mkdir(exist_ok=True)
for fonte in alvos:
    nb = jupytext.read(fonte, fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    destino = RAIZ / "Projeto3_Classificacao_Spam.ipynb" if fonte == HISTORIA else SAIDA / (fonte.stem + ".ipynb")
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}).execute()
        estado = "ok"
    except Exception as e:                       # salva mesmo assim, para ver onde parou
        estado = f"ERRO: {type(e).__name__}: {str(e)[:200]}"
    nbformat.write(nb, destino)
    print(f"{destino.relative_to(RAIZ)} | {time.time() - t0:.0f} s | {estado}", flush=True)
