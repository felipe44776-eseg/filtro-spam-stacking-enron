"""Viabilidade do TabPFN 3.5 local em CPU (tempo). Divisao aleatoria simples: nao e resultado."""
import os, time
os.environ.setdefault("TABPFN_NO_BROWSER", "1"); os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
t0 = time.time()
def lap(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, average_precision_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

df = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "dados" / "emails.csv")); y = df.pop("Prediction").values; df.pop("Email No.")
X = df.values.astype(np.float32)
lap(f"torch {torch.__version__} | threads {torch.get_num_threads()}")
for ver in [ModelVersion.V3_5_FAST, ModelVersion.V3_5]:
    for n in [300, 1000]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n, test_size=500, stratify=y, random_state=0)
        clf = TabPFNClassifier.create_default_for_version(ver, device="cpu", random_state=0)
        t = time.time(); clf.fit(Xtr, ytr); tf = time.time() - t
        t = time.time(); p = clf.predict_proba(Xte)[:, 1]; tp = time.time() - t
        lap(f"{ver.value:10s} treino={n:4d} fit={tf:6.1f}s predict(500)={tp:6.1f}s MCC={matthews_corrcoef(yte, p > .5):.3f} "
            f"PR-AUC={average_precision_score(yte, p):.3f}")
lap("fim")
