"""MCC e PR-AUC com previsoes out-of-fold reais do projeto."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (matthews_corrcoef, average_precision_score, roc_auc_score, f1_score,
                             accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve)

df = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "dados" / "emails.csv"))
y = df.pop("Prediction").values; df.pop("Email No.")
g = pd.util.hash_pandas_object(df, index=False).values
X = sparse.csr_matrix(df.values.astype(np.float32)); del df
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g))
tf = lambda: TfidfTransformer(sublinear_tf=True)
M = {"Reg. Logistica": make_pipeline(tf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
     "kNN cosseno": make_pipeline(tf(), KNeighborsClassifier(5, metric="cosine")),
     "Naive Bayes Compl.": ComplementNB()}
S = {}
for n, m in M.items():
    s = np.zeros(len(y))
    for tr, te in cv:
        s[te] = clone(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    S[n] = s

def linha(n, yh, s=None):
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(modelo=n, VP=tp, FP=fp, FN=fn, VN=tn, acuracia=accuracy_score(y, yh),
                precisao=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh),
                F1=f1_score(y, yh, zero_division=0), MCC=matthews_corrcoef(y, yh),
                PR_AUC=average_precision_score(y, s) if s is not None else np.nan,
                ROC_AUC=roc_auc_score(y, s) if s is not None else np.nan)

rng = np.random.default_rng(0)
lin = [linha(n, (s > 0.5).astype(int), s) for n, s in S.items()]
lin.append(linha("Dummy: tudo ham", np.zeros_like(y), np.full(len(y), y.mean())))
lin.append(linha("Chute aleatorio 29%", (rng.random(len(y)) < y.mean()).astype(int), rng.random(len(y))))
# filtro timido: so marca spam quando a LR tem certeza absoluta
lin.append(linha("LR timida (limiar 0,99)", (S["Reg. Logistica"] > 0.99).astype(int)))
# filtro paranoico: marca quase tudo como spam
lin.append(linha("LR paranoica (limiar 0,02)", (S["Reg. Logistica"] > 0.02).astype(int)))
T = pd.DataFrame(lin).set_index("modelo")
pd.set_option("display.width", 200)
print(T.round(3).to_string())

print("\nprevalencia de spam:", round(y.mean(), 3))
print("\n== Curva precisao-recall da Reg. Logistica (pontos escolhidos) ==")
p, r, t = precision_recall_curve(y, S["Reg. Logistica"])
for alvo in [0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]:
    i = np.where(r[:-1] >= alvo)[0][-1]
    fp_ham = ((S["Reg. Logistica"] >= t[i]) & (y == 0)).sum()
    print(f"recall>={alvo:.3f}  precisao={p[i]:.3f}  limiar={t[i]:.3f}  ham marcados como spam={fp_ham}")
