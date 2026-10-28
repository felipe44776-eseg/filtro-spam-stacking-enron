"""Como o texto vira números.

1. Contagem: cada uma das 3000 colunas do CSV conta quantas vezes a palavra aparece no texto pré-processado da
   AUEB, como *substring* e diferenciando maiúsculas (`texto.count(palavra)`). Foi assim que o CSV foi gerado:
   a reprodução bate em 5172 de 5172 e-mails.
2. TF-IDF com log (sublinear_tf): reduz o peso de palavras repetidas muitas vezes. Foi o detalhe que tirou a
   Regressão Logística de 0,923 para 0,961 de MCC.
3. NBFeatures (NBSVM): presença/ausência ponderada pela razão de chances do Naive Bayes.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer


def contar(texto: str, vocab) -> list[int]:
    """Vetor de contagens de um texto no vocabulário (substring, sensível a maiúsculas)."""
    return [texto.count(w) for w in vocab]


def tfidf() -> TfidfTransformer:
    return TfidfTransformer(sublinear_tf=True)


class NBFeatures(BaseEstimator, TransformerMixin):
    """NBSVM (Wang e Manning, 2012): binariza e pondera cada palavra pela log-razão do Naive Bayes."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        B = (X > 0).astype(np.float32)
        p = self.alpha + np.asarray(B[y == 1].sum(0)).ravel()
        q = self.alpha + np.asarray(B[y == 0].sum(0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        return self

    def transform(self, X):
        B = (X > 0).astype(np.float32)
        return B.multiply(self.r_).tocsr() if sparse.issparse(B) else B * self.r_
