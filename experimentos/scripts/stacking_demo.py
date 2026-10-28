"""Stacking feito a mao, para mostrar o mecanismo com os dados do projeto."""
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMClassifier

df = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "dados" / "emails.csv"))
y = df.pop("Prediction").values; ids = df.pop("Email No.").values
g = pd.util.hash_pandas_object(df, index=False).values
X = sparse.csr_matrix(df.values.astype(np.float32)); del df

BASE = {
    "LR":   make_pipeline(TfidfTransformer(sublinear_tf=True), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    "CNB":  ComplementNB(),
    "RF":   RandomForestClassifier(200, n_jobs=4, random_state=42),
    "LGBM": LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=4, random_state=42),
}
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g))

# Nivel 1: cada e-mail recebe a nota de modelos que NAO o viram no treino (out-of-fold)
Z = np.zeros((len(y), len(BASE)))
for j, (n, m) in enumerate(BASE.items()):
    t = time.time()
    for tr, te in cv:
        Z[te, j] = clone(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"{n:5s} OOF em {time.time() - t:5.1f}s")
Zd = pd.DataFrame(Z, columns=list(BASE))
H = (Zd > 0.5).astype(int)
E = H.ne(y, axis=0)

print("\n== MCC individual (limiar 0,5) e total de erros ==")
for n in BASE:
    print(f"{n:5s} MCC={matthews_corrcoef(y, H[n]):.3f}  erros={E[n].sum()}")
print(f"Media simples (votacao soft) MCC={matthews_corrcoef(y, (Zd.mean(1) > 0.5).astype(int)):.3f}")

print("\n== Correlacao entre as probabilidades ==")
print(Zd.corr().round(2).to_string())

print("\n== Erros em comum (quantos e-mails os dois erram juntos) ==")
print(pd.DataFrame({a: {b: int((E[a] & E[b]).sum()) for b in BASE} for a in BASE}).to_string())
k = E.sum(1)
print("\nE-mails errados por quantos modelos:", k.value_counts().sort_index().to_dict())

# Nivel 2: meta LR aprende quanto confiar em cada modelo
meta = LogisticRegression(max_iter=2000).fit(Z, y)
print("\n== Pesos aprendidos pela meta LR ==")
for n, c in zip(BASE, meta.coef_[0]):
    print(f"{n:5s} {c:+.2f}")
print(f"intercepto {meta.intercept_[0]:+.2f}")

# Exemplos de discordancia: onde os modelos de base brigam
pf = meta.predict_proba(Z)[:, 1]
discord = np.where((H.nunique(1) > 1).values)[0]
rng = np.random.default_rng(0)
amostra = rng.choice(discord, size=min(6, len(discord)), replace=False)
ex = Zd.iloc[amostra].round(2).copy()
ex.insert(0, "email", ids[amostra]); ex["media"] = Zd.iloc[amostra].mean(1).round(2)
ex["stacking"] = pf[amostra].round(2); ex["verdade"] = np.where(y[amostra] == 1, "spam", "ham")
print(f"\n== {len(discord)} e-mails com discordancia; amostra ==")
print(ex.to_string(index=False))
