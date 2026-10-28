"""Executa um notebook e salva as saidas mesmo se uma celula falhar."""
import sys, time, nbformat
from pathlib import Path
from nbclient import NotebookClient

arq = Path(sys.argv[1]); nb = nbformat.read(arq, as_version=4); t = time.time()
try:
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(arq.parent)}}).execute()
    print(f"OK em {time.time() - t:.0f}s")
except Exception as e:
    print(f"FALHOU em {time.time() - t:.0f}s: {type(e).__name__}\n{str(e)[-3000:]}")
finally:
    nbformat.write(nb, arq)
