"""Base externa: Enron1-6 da AUEB (Metsis, Androutsopoulos e Paliouras, 2006), contados com o vocabulário do CSV.

O CSV do enunciado é o Enron1. Enron2-6 são caixas de outras 5 pessoas, nunca vistas no treino: servem para
medir se o filtro generaliza. As cópias exatas de e-mails do CSV são removidas antes de avaliar.

Origem: http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enronN.tar.gz (cópia em dados/brutos_aueb/).
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
from scipy import sparse

from . import caminhos
from .dados import Base, carregar, hash_linha

URL_AUEB = "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{n}.tar.gz"
BASES_EXTERNAS = [f"Enron{k}" for k in range(2, 7)]

def _contar_lote(textos, vocab) -> np.ndarray:
    return np.array([[t.count(w) for w in vocab] for t in textos], dtype=np.int32)


def ler_tar(n: int, pasta: Path | None = None) -> tuple[list[str], np.ndarray, list[str]]:
    """(textos, rótulos, nomes) do enronN.tar.gz, na ordem alfabética dos caminhos (ham antes de spam).
    Os textos são lidos como latin-1, como no experimento original."""
    arq = Path(pasta or caminhos.BRUTOS_AUEB) / f"enron{n}.tar.gz"
    textos, rotulos, nomes = [], [], []
    with tarfile.open(arq, "r:gz") as tar:
        # só enronN/ham/*.txt e enronN/spam/*.txt (o Summary.txt da raiz fica de fora, como no experimento original)
        membros = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(".txt")
                          and m.name.split("/")[-2:-1] in (["ham"], ["spam"])), key=lambda m: m.name)
        for m in membros:
            # modo texto do Python converte \r\n em \n; replicamos para contar igual ao experimento original
            t = tar.extractfile(m).read().decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
            textos.append(t); rotulos.append(1 if "/spam/" in m.name else 0); nomes.append(m.name)
    return textos, np.array(rotulos), nomes


def vetorizar(textos, vocab, processos: int = 4) -> np.ndarray:
    """Contagem por substring de cada texto no vocabulário, em paralelo (cerca de 5 s por base com 4 processos).
    Usa joblib/loky: funciona em notebook e em script, sem o `if __name__ == "__main__"` que o Pool exige no Windows."""
    from joblib import Parallel, delayed
    vocab = list(vocab); textos = list(textos)
    lotes = [textos[i:i + 250] for i in range(0, len(textos), 250)]
    partes = Parallel(n_jobs=processos, backend="loky")(delayed(_contar_lote)(l, vocab) for l in lotes)
    return np.vstack(partes) if partes else np.zeros((0, len(vocab)), dtype=np.int32)


def gerar_vetores(destino: Path | None = None, base: Base | None = None, processos: int = 4) -> Path:
    """Refaz dados/enron_vetores.npz a partir dos .tar.gz (X1..X6 int32, y1..y6)."""
    base = base or carregar(); destino = Path(destino or caminhos.ENRON_VETORES); out = {}
    for n in range(1, 7):
        textos, y, _ = ler_tar(n)
        out[f"X{n}"] = vetorizar(textos, base.vocab, processos); out[f"y{n}"] = y
    np.savez_compressed(destino, **out)
    return destino


# ---------------------------------------------------------------------------------------------------------
# 3ª base: SpamAssassin public corpus (e-mails brutos, com cabeçalhos)
# ---------------------------------------------------------------------------------------------------------
URL_SPAMASSASSIN = "https://spamassassin.apache.org/old/publiccorpus/{arquivo}"
ARQUIVOS_SPAMASSASSIN = ["20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2", "20030228_hard_ham.tar.bz2",
                         "20030228_spam.tar.bz2", "20030228_spam_2.tar.bz2"]
PASTA_SPAMASSASSIN = caminhos.DADOS / "spamassassin"
VETORES_SPAMASSASSIN = caminhos.DADOS / "spamassassin_vetores.npz"


def baixar_spamassassin(pasta: Path | None = None) -> Path:
    """Baixa os 5 arquivos do corpus (cerca de 7 MB). O texto tem copyright dos remetentes: não vai para o repositório."""
    import urllib.request
    pasta = Path(pasta or PASTA_SPAMASSASSIN); pasta.mkdir(parents=True, exist_ok=True)
    for a in ARQUIVOS_SPAMASSASSIN:
        if not (pasta / a).exists():
            urllib.request.urlretrieve(URL_SPAMASSASSIN.format(arquivo=a), pasta / a)
    return pasta


def ler_spamassassin(pasta: Path | None = None) -> tuple[list[bytes], np.ndarray, list[str]]:
    """(mensagens brutas, rótulos, nomes). Spam = pastas spam e spam_2."""
    pasta = Path(pasta or PASTA_SPAMASSASSIN); brutos, y, nomes = [], [], []
    for a in ARQUIVOS_SPAMASSASSIN:
        with tarfile.open(pasta / a, "r:bz2") as tar:
            for m in sorted(tar.getmembers(), key=lambda m: m.name):
                partes = m.name.split("/")
                if not m.isfile() or partes[-1] == "cmds" or len(partes) < 2:
                    continue
                brutos.append(tar.extractfile(m).read()); y.append(1 if partes[-2].startswith("spam") else 0); nomes.append(m.name)
    return brutos, np.array(y), nomes


def _converter_lote(brutos, vocab) -> np.ndarray:
    from .conversao import para_aueb, vetor
    return np.array([vetor(para_aueb(b), vocab) for b in brutos], dtype=np.int32)


def vetorizar_brutos(brutos, vocab, processos: int = 4) -> np.ndarray:
    """Converte e-mails brutos para o formato da AUEB e conta o vocabulário (em paralelo)."""
    from joblib import Parallel, delayed
    vocab = list(vocab); lotes = [brutos[i:i + 250] for i in range(0, len(brutos), 250)]
    return np.vstack(Parallel(n_jobs=processos, backend="loky")(delayed(_converter_lote)(l, vocab) for l in lotes))


def gerar_vetores_spamassassin(base: Base | None = None) -> Path:
    base = base or carregar(); baixar_spamassassin()
    brutos, y, nomes = ler_spamassassin()
    np.savez_compressed(VETORES_SPAMASSASSIN, X=vetorizar_brutos(brutos, base.vocab), y=y, nomes=np.array(nomes))
    return VETORES_SPAMASSASSIN


def carregar_spamassassin(base: Base | None = None) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, grupos, mantidas): sem cópias exatas de linhas do CSV; grupos = hash do vetor (para o bootstrap);
    `mantidas` = máscara sobre o arquivo completo (para alinhar previsões feitas fora daqui, como as do TabPFN)."""
    base = base or carregar()
    if not VETORES_SPAMASSASSIN.exists():
        gerar_vetores_spamassassin(base)
    z = np.load(VETORES_SPAMASSASSIN); A = z["X"]
    h = [hash_linha(r) for r in A]
    novo = np.array([x not in base.hashes for x in h])
    g = np.array([int.from_bytes(x[:8], "little") for x in h])[novo]
    return sparse.csr_matrix(A[novo].astype(np.float32)), z["y"][novo], g, novo


def carregar_externo(base: Base | None = None, caminho: Path | None = None) -> tuple[dict, dict, dict]:
    """(Xext, yext, removidas): Enron2-6 sem as cópias exatas de e-mails do CSV. X esparsa float32."""
    base = base or carregar()
    z = np.load(caminho or caminhos.ENRON_VETORES)
    Xext, yext, removidas = {}, {}, {}
    for k in range(2, 7):
        A = z[f"X{k}"]
        novo = np.array([hash_linha(r) not in base.hashes for r in A])
        Xext[f"Enron{k}"] = sparse.csr_matrix(A[novo].astype(np.float32))
        yext[f"Enron{k}"] = z[f"y{k}"][novo]
        removidas[f"Enron{k}"] = int((~novo).sum())
    return Xext, yext, removidas
