import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
df = pd.read_csv(sys.argv[1]); y = df.pop("Prediction").values; df.pop("Email No."); X = df
# substring test: words containing 'ect' / 'hou'
ect_words = [c for c in X.columns if "ect" in c and c != "ect"]
hou_words = [c for c in X.columns if "hou" in c and c != "hou"]
print("cols containing 'ect':", ect_words[:15])
print("rows ect >= 1 + sum(ect-words):", round((X["ect"] >= 1 + X[ect_words].sum(1)).mean(), 3))
print("rows hou >= sum(hou-words):", round((X["hou"] >= X[hou_words].sum(1)).mean(), 3), hou_words[:10])
dup = X.duplicated().values; Xd, yd = X[~dup].values, y[~dup]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
f = lambda m, A, b: cross_val_score(m, A, b, cv=cv, scoring="f1").mean()
print("\nkNN + reducao (dedup, F1 CV5):")
for k in [50, 100, 300]:
    p = make_pipeline(TfidfTransformer(), TruncatedSVD(k, random_state=42), Normalizer(), KNeighborsClassifier(5))
    print(f"  tfidf+SVD{k}+kNN5: {f(p, Xd, yd):.3f}")
print("LR + chi2 selecao:")
for k in [100, 500, 1000, 3000]:
    p = make_pipeline(SelectKBest(chi2, k=k), TfidfTransformer(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))
    print(f"  chi2 k={k}: {f(p, Xd, yd):.3f}")
# duplicate leakage: naive split vs dedup split
m = make_pipeline(TfidfTransformer(), KNeighborsClassifier(1, metric="cosine"))
a = []; b = []
for s in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2, stratify=y, random_state=s)
    a.append(f1_score(yte, m.fit(Xtr, ytr).predict(Xte)))
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=s)
    b.append(f1_score(yte, m.fit(Xtr, ytr).predict(Xte)))
print(f"\n1-NN F1 split ingenuo={np.mean(a):.3f} vs dedup={np.mean(b):.3f} (10 seeds)")
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
keys = pd.util.hash_pandas_object(X, index=False).values
print("pct do teste ingenuo com gemeo exato no treino:", round(np.isin(keys[te], keys[tr]).mean()*100, 1))
