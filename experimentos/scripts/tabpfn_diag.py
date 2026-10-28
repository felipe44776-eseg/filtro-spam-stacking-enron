import os, time, sys
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
t = time.time(); lap = lambda s: print(f"[{time.time()-t:6.1f}s] {s}", flush=True)
import numpy as np; lap("numpy")
import torch; lap(f"torch {torch.__version__} threads={torch.get_num_threads()}")
from tabpfn import TabPFNClassifier; from tabpfn.constants import ModelVersion; lap("import tabpfn")
from tabpfn.settings import settings; lap(f"cache_dir={settings.tabpfn.model_cache_dir}")
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0); lap(f"criado model_path={clf.model_path} n_est={clf.n_estimators}")
X = np.random.rand(100, 10).astype(np.float32); y = (X[:, 0] > .5).astype(int)
clf.fit(X, y); lap("fit 100x10 (inclui download/carga dos pesos)")
clf.predict_proba(X); lap("predict 100x10")
