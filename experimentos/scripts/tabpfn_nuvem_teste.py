"""Viabilidade do TabPFN na nuvem (tabpfn-client): tempo e cota. Divisao aleatoria simples: nao e resultado.
Roda no ambiente separado .venv-tabpfn-nuvem. A chave vem de TABPFN_TOKEN e nao e impressa."""
import time
t0 = time.time()
def lap(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, average_precision_score
import tabpfn_client
from tabpfn_client import TabPFNClassifier

def uso():
    for f in ("get_api_usage",):
        for mod in (tabpfn_client, getattr(tabpfn_client, "config", None)):
            if mod is not None and hasattr(mod, f):
                try: return getattr(mod, f)()
                except Exception as e: return f"erro: {type(e).__name__}: {str(e)[:200]}"
    return "função de uso não encontrada"

df = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "dados" / "emails.csv")); y = df.pop("Prediction").values; df.pop("Email No.")
X = df.values.astype(np.float32)
lap(f"uso da API antes: {uso()}")
for n in [300, 1000]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, test_size=500, stratify=y, random_state=0)
    clf = TabPFNClassifier()
    t = time.time(); clf.fit(Xtr, ytr); tf = time.time() - t
    t = time.time(); p = clf.predict_proba(Xte)[:, 1]; tp = time.time() - t
    lap(f"nuvem ({getattr(clf, 'model_path', '?')}) treino={n:4d} fit={tf:6.1f}s predict(500)={tp:6.1f}s "
        f"MCC={matthews_corrcoef(yte, p > .5):.3f} PR-AUC={average_precision_score(yte, p):.3f}")
lap(f"uso da API depois: {uso()}")
lap("fim")
