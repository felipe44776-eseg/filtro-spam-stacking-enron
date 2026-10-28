import nbformat as nbf, sys
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("# Benchmark de viabilidade: torneio de modelos (rascunho técnico)\n"
   "Mesmos folds para todos os modelos (`StratifiedGroupKFold`, cópias idênticas no mesmo lado).")

code(r'''
import gc, time, platform, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sklearn, lightgbm, xgboost, catboost
SEED = 42
print(platform.python_version(), "sklearn", sklearn.__version__, "lgbm", lightgbm.__version__,
      "xgb", xgboost.__version__, "catboost", catboost.__version__)
''')

code(r'''
CSV = next(p for p in [Path("../../dados/emails.csv"), Path("emails.csv"), Path("emails (1).csv"), Path("../../emails (1).csv")] if p.exists())
df = pd.read_csv(CSV); y = df.pop("Prediction").values; df.pop("Email No.")
grupos = pd.util.hash_pandas_object(df, index=False).values
X = sparse.csr_matrix(df.values.astype(np.float32))   # 94% de zeros: ~16x menos memoria que denso
del df; gc.collect()
print(X.shape, f"esparsa {X.data.nbytes / 1e6:.0f} MB", "spam%", round(y.mean() * 100, 1), "grupos", len(set(grupos)))
''')

code(r'''
class NBFeatures(BaseEstimator, TransformerMixin):
    """NBSVM (Wang & Manning 2012): binariza e pondera cada palavra pela log-razao do Naive Bayes."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        B = (X > 0).astype(np.float32)
        p = self.alpha + np.asarray(B[y == 1].sum(0)).ravel(); q = self.alpha + np.asarray(B[y == 0].sum(0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        return self
    def transform(self, X):
        B = (X > 0).astype(np.float32)
        return B.multiply(self.r_).tocsr() if sparse.issparse(B) else B * self.r_

N_JOBS = 4
denso = lambda: FunctionTransformer(lambda A: A.toarray() if sparse.issparse(A) else A, accept_sparse=True)
tfidf = lambda: TfidfTransformer(sublinear_tf=True)
lr_tfidf = lambda: make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))
base_ens = lambda: [("lr", lr_tfidf()), ("cnb", ComplementNB()),
                    ("rf", RandomForestClassifier(200, n_jobs=N_JOBS, random_state=SEED)),
                    ("lgbm", LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=N_JOBS, random_state=SEED))]
MODELOS = {
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
  "Random Forest":            lambda: RandomForestClassifier(400, n_jobs=N_JOBS, class_weight="balanced", random_state=SEED),
  "Extra Trees":              lambda: ExtraTreesClassifier(400, n_jobs=N_JOBS, class_weight="balanced", random_state=SEED),
  "HistGradientBoosting":     lambda: make_pipeline(denso(), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)),
  "LightGBM":                 lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=N_JOBS, verbose=-1, random_state=SEED),
  "XGBoost":                  lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=N_JOBS, random_state=SEED),
  "CatBoost":                 lambda: make_pipeline(denso(), CatBoostClassifier(iterations=400, learning_rate=0.1, thread_count=N_JOBS, verbose=0, random_seed=SEED, allow_writing_files=False)),
  "MLP (rede neural)":        lambda: make_pipeline(tfidf(), MLPClassifier((128,), early_stopping=True, max_iter=200, random_state=SEED)),
  "Votacao (LR+CNB+RF+LGBM)": lambda: VotingClassifier(base_ens(), voting="soft"),
  "Stacking (meta LR)":       lambda: StackingClassifier(base_ens(), final_estimator=LogisticRegression(max_iter=2000), cv=5),
}

def escore(m, A):
    return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)

def avaliar(nome, seed):
    linhas = []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grupos)):
        m = MODELOS[nome](); t0 = time.time(); m.fit(X[tr], y[tr]); t1 = time.time()
        yh = m.predict(X[te]); s = escore(m, X[te]); t2 = time.time()
        linhas.append(dict(modelo=nome, seed=seed, fold=f,
            mcc=matthews_corrcoef(y[te], yh), pr_auc=average_precision_score(y[te], s), f1=f1_score(y[te], yh),
            fp_ham=((yh == 1) & (y[te] == 0)).sum() / (y[te] == 0).sum(),
            fn_spam=((yh == 0) & (y[te] == 1)).sum() / (y[te] == 1).sum(),
            fit_s=t1 - t0, pred_s=t2 - t1, n_tr=len(tr), n_te=len(te)))
        del m; gc.collect()
    return linhas

def rodar(arquivo, nomes, seeds):
    """Salva a cada modelo e retoma de onde parou se a execucao cair."""
    feitos = pd.read_csv(arquivo, sep=";", decimal=",") if Path(arquivo).exists() else pd.DataFrame(columns=["modelo", "seed"])
    for seed in seeds:
        for nome in nomes:
            if ((feitos.modelo == nome) & (feitos.seed == seed)).any():
                continue
            t = time.time(); novo = pd.DataFrame(avaliar(nome, seed))
            novo.to_csv(arquivo, sep=";", decimal=",", index=False, mode="a", header=not Path(arquivo).exists())
            print(f"seed {seed} | {nome:28s} {time.time() - t:6.1f}s", flush=True)
    return pd.read_csv(arquivo, sep=";", decimal=",")
''')

md("## Rodada 1: triagem (todos os modelos, 1×5 folds)")
code(r'''
R1 = rodar("bench_rodada1.csv", list(MODELOS), [SEED])
tab1 = (R1.groupby("modelo")
          .agg(MCC=("mcc", "mean"), MCC_dp=("mcc", "std"), PR_AUC=("pr_auc", "mean"), F1=("f1", "mean"),
               FP_ham=("fp_ham", "mean"), FN_spam=("fn_spam", "mean"), treino_s=("fit_s", "mean"))
          .sort_values("MCC", ascending=False).round(3))
tab1
''')

md("## Rodada 2: top 5 em validação cruzada repetida (3×5) + teste t corrigido (Nadeau-Bengio) com Holm")
code(r'''
from scipy import stats
TOP = list(tab1.index[:5])
R2 = pd.concat([R1[R1.modelo.isin(TOP)], rodar("bench_rodada2.csv", TOP, [7, 2026])], ignore_index=True)
R2 = R2[R2.modelo.isin(TOP)]

def t_corrigido(a, b, n_tr, n_te):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)

P = R2.pivot_table(index=["seed", "fold"], columns="modelo", values="mcc")
melhor = P.mean().idxmax(); n_tr, n_te = R2.n_tr.mean(), R2.n_te.mean()
comp = pd.DataFrame([dict(contra=o, dif_MCC=d, t=t, p=p)
                     for o in P.columns.drop(melhor)
                     for d, t, p in [t_corrigido(P[melhor], P[o], n_tr, n_te)]]).sort_values("p")
m = len(comp)
comp["p_holm"] = np.minimum(1, np.maximum.accumulate([(m - i) * p for i, p in enumerate(comp.p.values)]))
print("melhor (media de 15 folds):", melhor, round(P[melhor].mean(), 3))
print(P.mean().sort_values(ascending=False).round(4).to_string())
comp.round(4)
''')

code(r'''
import baycomp
seg = P.drop(columns=melhor).mean().idxmax()
pr = baycomp.two_on_single(P[melhor].values, P[seg].values, rope=0.01, runs=3)
print(f"Bayesiano correlacionado {melhor} vs {seg}: "
      f"P({melhor} melhor)={pr[0]:.3f} | P(equivalentes, |dif MCC|<0,01)={pr[1]:.3f} | P({seg} melhor)={pr[2]:.3f}")
''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1]); print("ok", sys.argv[1])
