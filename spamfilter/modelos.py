"""Catálogo de modelos: o torneio de 22, os 13 de base do experimento 02, o espaço de busca do Stacking 7
e os filtros finais (Especialista e Stacking 7 padrão; o Stacking 8 acrescenta o TabPFN).

Todo modelo que usa sorteio recebe SEED = 42, para que os resultados se repitam.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV, RidgeClassifier,
                                  SGDClassifier)
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .dados import SEED
from .representacao import NBFeatures, tfidf


def nota(m, A) -> np.ndarray:
    """Nota contínua de spam: probabilidade quando existe, senão a distância à fronteira (LinearSVC, Ridge)."""
    return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)


# ---------------------------------------------------------------------------------------------------------
# Construtores das 7 famílias do Stacking 7 (+ LinearSVC, a linha de base)
# ---------------------------------------------------------------------------------------------------------
def lr(C, cw, n_jobs=4):
    return make_pipeline(tfidf(), LogisticRegression(C=C, class_weight=cw, max_iter=5000))

def svm(k, C, n_jobs=4):
    return make_pipeline(tfidf(), TruncatedSVD(k, random_state=SEED), Normalizer(), SVC(C=C, class_weight="balanced"))

def knn(k, w, n_jobs=4):
    return make_pipeline(tfidf(), KNeighborsClassifier(k, weights=w, metric="cosine"))

def mlp(h, a, n_jobs=4):
    return make_pipeline(tfidf(), MLPClassifier(h, alpha=a, early_stopping=True, max_iter=200, random_state=SEED))

def cnb(a, norm, usa_tf, n_jobs=4):
    return make_pipeline(tfidf(), ComplementNB(alpha=a, norm=norm)) if usa_tf else ComplementNB(alpha=a, norm=norm)

def et(n, mf, msl, cw, n_jobs=4):
    return ExtraTreesClassifier(n, max_features=mf, min_samples_leaf=msl, class_weight=cw, n_jobs=n_jobs, random_state=SEED)

def lgbm(n, eta, folhas, col=1.0, sub=1.0, mcs=20, n_jobs=4):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=n, learning_rate=eta, num_leaves=folhas, colsample_bytree=col, subsample=sub,
                          subsample_freq=1 if sub < 1 else 0, min_child_samples=mcs, n_jobs=n_jobs, verbose=-1, random_state=SEED)

def lsvc(C, cw, n_jobs=4):
    return make_pipeline(tfidf(), LinearSVC(C=C, class_weight=cw))


# Espaço de busca do algoritmo genético. A posição 0 de cada família é a configuração padrão (experimento 02).
# A ORDEM das famílias define o índice f{N} dos arquivos de cache: não reordenar.
ESPACO = {
    "Reg. Logística": [(lr, dict(C=C, cw=cw)) for C, cw in
                       [(10, "balanced"), (1, "balanced"), (3, "balanced"), (30, "balanced"), (100, "balanced"),
                        (10, None), (1, None), (100, None)]],
    "SVM RBF":        [(svm, dict(k=k, C=C)) for k, C in
                       [(300, 10), (300, 3), (300, 30), (150, 10), (150, 30), (500, 10), (300, 1), (500, 30)]],
    "kNN cosseno":    [(knn, dict(k=k, w=w)) for k, w in
                       [(5, "uniform"), (3, "uniform"), (7, "uniform"), (11, "uniform"), (15, "uniform"),
                        (5, "distance"), (11, "distance"), (21, "distance")]],
    "MLP":            [(mlp, dict(h=h, a=a)) for h, a in
                       [((128,), 1e-4), ((64,), 1e-4), ((256,), 1e-4), ((128,), 1e-3), ((128,), 1e-2), ((256, 128), 1e-4)]],
    "Naive Bayes":    [(cnb, dict(a=a, norm=n, usa_tf=t)) for a, n, t in
                       [(1, False, False), (0.1, False, False), (0.3, False, False), (3, False, False),
                        (1, True, False), (0.1, True, False), (0.1, False, True), (1, False, True)]],
    "Extra Trees":    [(et, dict(n=n, mf=mf, msl=msl, cw=cw)) for n, mf, msl, cw in
                       [(400, "sqrt", 1, "balanced"), (400, "sqrt", 1, None), (400, "log2", 1, "balanced"),
                        (400, 0.05, 1, "balanced"), (400, "sqrt", 2, "balanced"), (800, "sqrt", 1, "balanced")]],
    "LightGBM":       [(lgbm, kw) for kw in
                       [dict(n=400, eta=0.05, folhas=31), dict(n=200, eta=0.1, folhas=31), dict(n=800, eta=0.03, folhas=31),
                        dict(n=400, eta=0.05, folhas=15), dict(n=400, eta=0.05, folhas=63),
                        dict(n=400, eta=0.05, folhas=31, col=0.3), dict(n=800, eta=0.05, folhas=15, col=0.3, sub=0.8),
                        dict(n=400, eta=0.05, folhas=31, mcs=5)]],
}
LINEARSVC = [(lsvc, dict(C=C, cw=cw)) for C, cw in
             [(1, "balanced"), (0.1, "balanced"), (0.3, "balanced"), (3, "balanced"), (10, "balanced"),
              (1, None), (0.3, None), (3, None)]]
META_C = [1.0, 0.03, 0.1, 0.3, 3.0, 10.0]

FAMILIAS = list(ESPACO)                       # as 7 famílias, na ordem dos genes
TODOS = {**ESPACO, "LinearSVC": LINEARSVC}
FID = {f: i for i, f in enumerate(TODOS)}     # índice da família no nome do arquivo de cache
CARD = [len(ESPACO[f]) for f in FAMILIAS] + [len(META_C)]   # opções por gene (7 famílias + C do meta-modelo)

# Os filtros finais (D8 do registro de decisões): um índice por família + o índice do C do meta-modelo.
STACKING7_PADRAO = (0, 0, 0, 0, 0, 0, 0, 0)          # Stacking 7 padrão (base do Stacking 8)
STACKING7_ESPECIALISTA = (1, 7, 1, 1, 5, 5, 4, 5)    # escolhido pelo algoritmo genético (#5 do experimento 03)
LINEARSVC_PADRAO = 0                                 # C=1, balanced: a linha de base
LINEARSVC_OTIMIZADO = 5                              # C=1, sem balanceamento: o melhor na busca


def construir(familia: str, k: int, n_jobs: int = 4):
    f, kw = TODOS[familia][k]
    return f(**kw, n_jobs=n_jobs)


def descrever(familia: str, k: int) -> str:
    return ", ".join(f"{a}={b}" for a, b in TODOS[familia][k][1].items())


def descrever_stacking(cfg) -> dict:
    """{família: parâmetros} de uma configuração do Stacking 7."""
    d = {fam: descrever(fam, cfg[i]) for i, fam in enumerate(FAMILIAS)}
    d["meta-modelo"] = f"C={META_C[cfg[-1]]}"
    return d


def meta(C: float = 1.0):
    """Meta-modelo do stacking: Regressão Logística sobre as 7 notas padronizadas."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


# ---------------------------------------------------------------------------------------------------------
# Torneio de 22 modelos independentes (experimento 01)
# ---------------------------------------------------------------------------------------------------------
def _denso():
    return FunctionTransformer(lambda A: A.toarray() if sparse.issparse(A) else A, accept_sparse=True)


def torneio(n_jobs: int = 4) -> dict:
    """nome -> construtor. Os nomes seguem o arquivo bench_rodada1.csv do experimento 01.
    Observação: o "Stacking (meta LR)" deste torneio usa o StackingClassifier do sklearn, cuja CV interna não
    separa por grupo. A partir do experimento 02, o stacking usa CV interna por grupo (motor.py)."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    lr_tfidf = lambda: make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))
    base_ens = lambda: [("lr", lr_tfidf()), ("cnb", ComplementNB()),
                        ("rf", RandomForestClassifier(200, n_jobs=n_jobs, random_state=SEED)),
                        ("lgbm", LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=n_jobs, random_state=SEED))]
    return {
        "Dummy (maioria)":          lambda: DummyClassifier(strategy="prior"),
        "kNN TF-IDF cosseno":       lambda: make_pipeline(tfidf(), KNeighborsClassifier(5, metric="cosine")),
        "kNN SVD100":               lambda: make_pipeline(tfidf(), TruncatedSVD(100, random_state=SEED), Normalizer(), KNeighborsClassifier(5)),
        "Naive Bayes Multinomial":  lambda: MultinomialNB(),
        "Naive Bayes Complement":   lambda: ComplementNB(),
        "Naive Bayes Bernoulli":    lambda: BernoulliNB(binarize=0.5),
        "Reg. Logistica TF-IDF":    lr_tfidf,
        "LinearSVC TF-IDF":         lambda: make_pipeline(tfidf(), LinearSVC(C=1, class_weight="balanced")),
        "SGD (huber) TF-IDF":       lambda: make_pipeline(tfidf(), SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=SEED)),
        "Ridge TF-IDF":             lambda: make_pipeline(tfidf(), RidgeClassifier(class_weight="balanced")),
        "NBSVM":                    lambda: make_pipeline(NBFeatures(), LogisticRegression(C=1, max_iter=3000)),
        "SVM RBF SVD300":           lambda: make_pipeline(tfidf(), TruncatedSVD(300, random_state=SEED), Normalizer(), SVC(C=10, class_weight="balanced")),
        "Arvore de Decisao":        lambda: DecisionTreeClassifier(min_samples_leaf=2, class_weight="balanced", random_state=SEED),
        "Random Forest":            lambda: RandomForestClassifier(400, n_jobs=n_jobs, class_weight="balanced", random_state=SEED),
        "Extra Trees":              lambda: ExtraTreesClassifier(400, n_jobs=n_jobs, class_weight="balanced", random_state=SEED),
        "HistGradientBoosting":     lambda: make_pipeline(_denso(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)),
        "LightGBM":                 lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=n_jobs, verbose=-1, random_state=SEED),
        "XGBoost":                  lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=n_jobs, random_state=SEED),
        "CatBoost":                 lambda: make_pipeline(_denso(), CatBoostClassifier(iterations=400, learning_rate=0.1, thread_count=n_jobs, verbose=0,
                                                                                       random_seed=SEED, allow_writing_files=False)),
        "MLP (rede neural)":        lambda: make_pipeline(tfidf(), MLPClassifier((128,), early_stopping=True, max_iter=200, random_state=SEED)),
        "Votacao (LR+CNB+RF+LGBM)": lambda: VotingClassifier(base_ens(), voting="soft"),
        "Stacking (meta LR)":       lambda: StackingClassifier(base_ens(), final_estimator=LogisticRegression(max_iter=2000), cv=5),
    }


# Nome de exibição (com acentos) de cada modelo do torneio.
NOMES_TORNEIO = {"Reg. Logistica TF-IDF": "Reg. Logística TF-IDF", "Arvore de Decisao": "Árvore de Decisão",
                 "Votacao (LR+CNB+RF+LGBM)": "Votação (LR+CNB+RF+LGBM)"}

# Família de cada modelo do torneio (para os gráficos).
FAMILIA_TORNEIO = {
    "Dummy (maioria)": "referência", "kNN TF-IDF cosseno": "vizinhos", "kNN SVD100": "vizinhos",
    "Naive Bayes Multinomial": "probabilística", "Naive Bayes Complement": "probabilística", "Naive Bayes Bernoulli": "probabilística",
    "Reg. Logistica TF-IDF": "linear", "LinearSVC TF-IDF": "linear", "SGD (huber) TF-IDF": "linear", "Ridge TF-IDF": "linear",
    "NBSVM": "linear", "SVM RBF SVD300": "kernel", "Arvore de Decisao": "árvore", "Random Forest": "bagging",
    "Extra Trees": "bagging", "HistGradientBoosting": "boosting", "LightGBM": "boosting", "XGBoost": "boosting",
    "CatBoost": "boosting", "MLP (rede neural)": "rede neural", "Votacao (LR+CNB+RF+LGBM)": "combinação",
    "Stacking (meta LR)": "combinação",
}


# ---------------------------------------------------------------------------------------------------------
# Experimento 02: 13 modelos de base e 6 formas de combinar
# ---------------------------------------------------------------------------------------------------------
def base13(n_jobs: int = 4) -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "Reg. Logística":     lambda: make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
        "LinearSVC":          lambda: make_pipeline(tfidf(), LinearSVC(C=1, class_weight="balanced")),
        "SGD (huber)":        lambda: make_pipeline(tfidf(), SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=SEED)),
        "Ridge":              lambda: make_pipeline(tfidf(), RidgeClassifier(class_weight="balanced")),
        "NBSVM":              lambda: make_pipeline(NBFeatures(), LogisticRegression(C=1, max_iter=3000)),
        "SVM RBF (SVD300)":   lambda: make_pipeline(tfidf(), TruncatedSVD(300, random_state=SEED), Normalizer(), SVC(C=10, class_weight="balanced")),
        "kNN cosseno":        lambda: make_pipeline(tfidf(), KNeighborsClassifier(5, metric="cosine")),
        "MLP (rede neural)":  lambda: make_pipeline(tfidf(), MLPClassifier((128,), early_stopping=True, max_iter=200, random_state=SEED)),
        "Naive Bayes Compl.": lambda: ComplementNB(),
        "Random Forest":      lambda: RandomForestClassifier(400, n_jobs=n_jobs, class_weight="balanced", random_state=SEED),
        "Extra Trees":        lambda: ExtraTreesClassifier(400, n_jobs=n_jobs, class_weight="balanced", random_state=SEED),
        "LightGBM":           lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=n_jobs, verbose=-1, random_state=SEED),
        "XGBoost":            lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=n_jobs, random_state=SEED),
    }


def platt(Ztr, ytr, Zte):
    """Põe cada nota na escala de probabilidade, calibrando só com o treino (necessário para média e Caruana)."""
    Ptr, Pte = np.zeros_like(Ztr), np.zeros_like(Zte)
    for j in range(Ztr.shape[1]):
        c = LogisticRegression(C=1e4, max_iter=1000).fit(Ztr[:, [j]], ytr)
        Ptr[:, j], Pte[:, j] = c.predict_proba(Ztr[:, [j]])[:, 1], c.predict_proba(Zte[:, [j]])[:, 1]
    return Ptr, Pte


def meta_lr(l1: bool = False):
    """Combinador: meta Regressão Logística (com L1, zera os modelos inúteis). Devolve f -> (nota, pesos)."""
    def f(Ztr, ytr, Zte, Ptr, Pte):
        est = (LogisticRegressionCV(Cs=10, l1_ratios=(1.0,), solver="saga", scoring="neg_log_loss",
                                    max_iter=5000, use_legacy_attributes=False) if l1 else LogisticRegression(max_iter=2000))
        m = make_pipeline(StandardScaler(), est).fit(Ztr, ytr)
        return m.predict_proba(Zte)[:, 1], m[-1].coef_.ravel()
    return f


def media(Ztr, ytr, Zte, Ptr, Pte):
    """Combinador: votação (média simples das probabilidades calibradas)."""
    return Pte.mean(1), np.full(Pte.shape[1], 1 / Pte.shape[1])


def caruana(Ztr, ytr, Zte, Ptr, Pte, n_iter: int = 50):
    """Combinador: seleção gulosa de Caruana et al. (2004), com reposição."""
    soma, escolha = np.zeros(len(ytr)), []
    for k in range(1, n_iter + 1):
        perdas = [log_loss(ytr, np.clip((soma + Ptr[:, j]) / k, 1e-6, 1 - 1e-6)) for j in range(Ptr.shape[1])]
        j = int(np.argmin(perdas)); escolha.append(j); soma += Ptr[:, j]
    w = np.bincount(escolha, minlength=Ptr.shape[1]) / n_iter
    return Pte @ w, w
