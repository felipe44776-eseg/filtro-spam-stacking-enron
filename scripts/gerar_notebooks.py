"""Gera os notebooks a partir dos scripts (formato # %%) e executa cada um, salvando as saídas.

    python scripts/gerar_notebooks.py            # todos
    python scripts/gerar_notebooks.py 03 05      # só alguns

Os notebooks NÃO são editados à mão: a fonte é o script. Por padrão, tudo usa os caches gravados (minutos).
Cada notebook ganha, logo depois do título, o botão "Abrir no Colab" e a célula que prepara o Google Colab.
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
REPO = "felipe44776-eseg/filtro-spam-stacking-enron"

COLAB_MD = """[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{repo}/blob/main/{caminho})

**No Google Colab:** rode tudo (*Ambiente de execução → Executar tudo*). Na primeira vez, a célula abaixo baixa o
projeto, instala os pacotes e **reinicia o ambiente**; quando o Colab reconectar, use *Executar tudo* de novo.
No computador, ela não faz nada (siga o [INSTALACAO.md](https://github.com/{repo}/blob/main/INSTALACAO.md))."""

COLAB_CODIGO = """# Google Colab: prepara o ambiente na primeira execução. No computador, esta célula não faz nada.
import os, subprocess, sys
if "google.colab" in sys.modules:
    PROJETO = "/content/filtro-spam-stacking-enron"
    if not os.path.isdir(PROJETO):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/{repo}.git", PROJETO], check=True)
    os.chdir(PROJETO)
    sys.path.insert(0, PROJETO)
    MARCA = "/content/.spamfilter_instalado"
    if not os.path.exists(MARCA):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
        open(MARCA, "w").close()
        print("Pacotes instalados. O Colab vai reiniciar o ambiente: quando reconectar, use Executar tudo de novo.")
        os.kill(os.getpid(), 9)"""


def preparar_colab(nb, caminho_no_repo: str):
    md = nbformat.v4.new_markdown_cell(COLAB_MD.format(repo=REPO, caminho=caminho_no_repo))
    cod = nbformat.v4.new_code_cell(COLAB_CODIGO.format(repo=REPO))
    md.metadata["tags"] = cod.metadata["tags"] = ["colab"]
    nb.cells[1:1] = [md, cod]


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
    preparar_colab(nb, destino.relative_to(RAIZ).as_posix())
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}).execute()
        estado = "ok"
    except Exception as e:                       # salva mesmo assim, para ver onde parou
        estado = f"ERRO: {type(e).__name__}: {str(e)[:200]}"
    nbformat.write(nb, destino)
    print(f"{destino.relative_to(RAIZ)} | {time.time() - t0:.0f} s | {estado}", flush=True)
