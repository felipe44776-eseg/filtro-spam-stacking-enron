import os, sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, average_precision_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import torch; torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
df = pd.read_csv(sys.argv[1]); y = df.pop("Prediction").values; df.pop("Email No."); X = df.values.astype(np.float32)
grp = pd.util.hash_pandas_object(df, index=False).values
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, grp))
n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else len(tr)
tr = np.random.RandomState(0).permutation(tr)[:n_tr]
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=42)
m = make_pipeline(SelectKBest(chi2, k=500), clf)
t0 = time.time(); m.fit(X[tr], y[tr]); t1 = time.time()
p = m.predict_proba(X[te])[:, 1]; t2 = time.time()
yh = (p >= 0.5).astype(int)
print(f"TabPFN v2 | treino={len(tr)} teste={len(te)} | fit={t1-t0:.1f}s predict={t2-t1:.1f}s | MCC={matthews_corrcoef(y[te], yh):.3f} PR-AUC={average_precision_score(y[te], p):.3f} FPR_ham={((yh==1)&(y[te]==0)).sum()/(y[te]==0).sum():.3f}", flush=True)
