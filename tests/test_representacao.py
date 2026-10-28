"""T3.2 e T3.7: a contagem por substring reproduz o CSV; TF-IDF e NBSVM."""
import random
from collections import Counter

import numpy as np
from scipy import sparse

from spamfilter import externo, representacao


def test_contagem_por_substring_reproduz_o_csv(base):
    """300 e-mails sorteados do Enron1 original, contados pelo pacote, aparecem idênticos no CSV."""
    textos, _, _ = externo.ler_tar(1)
    assert len(textos) == 5172
    linhas_csv = Counter(r.astype(np.int64).tobytes() for r in base.X.toarray())
    random.seed(0)
    amostra = random.sample(textos, 300)
    achados = sum(np.array(representacao.contar(t, base.vocab), dtype=np.int64).tobytes() in linhas_csv for t in amostra)
    assert achados == 300


def test_contagem_diferencia_maiusculas():
    # "ree" aparece dentro dos dois "free", mas não em "FREE"
    assert representacao.contar("Subject: free FREE free", ["free", "FREE", "ree"]) == [2, 1, 2]


def test_tfidf_usa_log():
    assert representacao.tfidf().sublinear_tf is True


def test_nbfeatures_pondera_pela_classe():
    X = sparse.csr_matrix(np.array([[3, 0], [1, 0], [0, 2], [0, 1]], dtype=np.float32))
    y = np.array([1, 1, 0, 0])
    nb = representacao.NBFeatures().fit(X, y)
    assert nb.r_[0] > 0 > nb.r_[1]                      # coluna 0 só no spam; coluna 1 só no e-mail real
    T = nb.transform(X).toarray()
    assert set(np.unique(np.abs(T[T != 0]).round(5))) <= set(np.abs(nb.r_).round(5))   # binariza e pondera
