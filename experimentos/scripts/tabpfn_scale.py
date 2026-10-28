import os, sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import matthews_corrcoef, average_precision_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
csv, k, n_est, n_tr, n_te = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
df = pd.read_csv(csv); y = df.pop("Prediction").values; df.pop("Email No."); X = df.values.astype(np.float32)
grp = pd.util.hash_pandas_object(df, index=False).values
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, grp))
rs = np.random.RandomState(0); tr = rs.permutation(tr)[:n_tr]; te = rs.permutation(te)[:n_te]
sel = SelectKBest(chi2, k=k).fit(X[tr], y[tr]); Xtr, Xte = sel.transform(X[tr]), sel.transform(X[te])
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=n_est, random_state=42)
t0 = time.time(); clf.fit(Xtr, y[tr]); t1 = time.time(); p = clf.predict_proba(Xte)[:, 1]; t2 = time.time()
yh = (p >= .5).astype(int)
print(f"k={k} n_est={n_est} treino={len(tr)} teste={len(te)} | fit={t1-t0:.1f}s predict={t2-t1:.1f}s | MCC={matthews_corrcoef(y[te], yh):.3f} PR-AUC={average_precision_score(y[te], p):.3f}", flush=True)
