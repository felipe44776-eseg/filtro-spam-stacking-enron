"""Carga da base, grupos de cópias (linhas de contagem iguais) e divisões sem vazamento.

Regra central: e-mails com o mesmo vetor de contagens formam um grupo e ficam sempre do mesmo lado
(treino ou teste). Sem isso, uma cópia no treino "ensina" a resposta da cópia no teste.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

from . import caminhos

SEED = 42           # semente dos modelos e das CVs internas
SEED_TESTE = 123    # semente da base de teste trancada (20%)


@dataclass(frozen=True)
class Base:
    """Os 5172 e-mails: X (contagens, esparsa float32), y (1 = spam), g (grupo = hash da linha)."""
    X: sparse.csr_matrix
    y: np.ndarray
    g: np.ndarray
    vocab: tuple
    hashes: frozenset = field(repr=False)   # md5 de cada linha; usado para tirar cópias da base externa

    @property
    def n(self) -> int:
        return self.X.shape[0]


def hash_linha(linha: np.ndarray) -> bytes:
    """Identidade de um vetor de contagens (o mesmo critério para o CSV e para a base externa)."""
    return hashlib.md5(np.ascontiguousarray(linha, dtype=np.int64).tobytes()).digest()


@lru_cache(maxsize=2)
def carregar(caminho: str | Path | None = None) -> Base:
    df = pd.read_csv(caminho or caminhos.CSV)
    y = df.pop("Prediction").to_numpy()
    df.pop("Email No.")
    g = pd.util.hash_pandas_object(df, index=False).to_numpy()
    A = np.ascontiguousarray(df.to_numpy(), dtype=np.int64)
    hashes = frozenset(hash_linha(r) for r in A)
    X = sparse.csr_matrix(A.astype(np.float32))
    return Base(X=X, y=y, g=g, vocab=tuple(df.columns), hashes=hashes)


def divisoes(y: np.ndarray, g: np.ndarray, semente: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds estratificados por classe e separados por grupo."""
    cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=semente)
    return list(cv.split(np.zeros(len(y)), y, g))


def indices_teste(base: Base | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(desenvolvimento 80%, teste trancado 20%): o primeiro fold de SEED_TESTE. Recalculado, não lido de arquivo."""
    base = base or carregar()
    dev, teste = divisoes(base.y, base.g, SEED_TESTE)[0]
    assert not set(base.g[dev]) & set(base.g[teste]), "texto repetido dos dois lados"
    return dev, teste


def resumo_duplicatas(base: Base | None = None) -> dict:
    """Quantas linhas são cópias de outra e de que classe elas são."""
    base = base or carregar()
    s = pd.Series(base.g)
    repetida = s.duplicated(keep="first").to_numpy()   # cópias além da primeira ocorrência
    return dict(n_emails=base.n, textos_distintos=int(s.nunique()), duplicatas_excedentes=int(repetida.sum()),
                fracao_duplicatas_email_real=float((base.y[repetida] == 0).mean()))
