"""Os filtros treinados (resultados/modelos/*.joblib) ficam fora do git, na página Releases do GitHub.

    from spamfilter.publicados import garantir
    garantir("especialista")          # baixa da Release v1.0 se faltar e confere o sha256

Usado pelo notebook-história, pela demo e pelo Google Colab. Sem internet, os mesmos arquivos saem de
`python scripts/09_modelos_finais.py` (cerca de 5 min).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from . import caminhos

RELEASE = "https://github.com/felipe44776-eseg/filtro-spam-stacking-enron/releases/download/v1.0/"
SHA256 = {
    "especialista": "03a412f50cc0af13915fc48bd7ce549b3fa2b50ec3396b679a4e815153c8c5e6",
    "stacking7_padrao": "1e9fba3c584e6b0569553f1504a33d0d85d58faa9c8d48e458ef8c6127f54862",
    "generalista_stacking8": "c6c9da9dc7f72642f16ec0008ed2068d366caebfabc27d3917fff42d6783fbfd",
    "linha_de_base": "09b4094b668947f67244b745030996e374c1f8dbaee263b2451e87c8120ea3eb",
}


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def garantir(nome: str, pasta: Path | None = None, log=print) -> Path:
    """Caminho do filtro `nome`. Se o arquivo não existir, baixa da Release e confere o sha256."""
    pasta = Path(pasta or caminhos.RESULTADOS / "modelos")
    arq = pasta / f"{nome}.joblib"
    if arq.exists():
        return arq
    pasta.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_suffix(".parcial")
    log(f"baixando {arq.name} da Release v1.0 do GitHub...")
    with urllib.request.urlopen(RELEASE + arq.name, timeout=120) as r, open(tmp, "wb") as f:
        while bloco := r.read(1 << 20):
            f.write(bloco)
    if _sha256(tmp) != SHA256[nome]:
        tmp.unlink()
        raise RuntimeError(f"{arq.name}: o sha256 baixado não confere com o publicado")
    tmp.replace(arq)
    return arq
