"""TabPFN-3.5-Plus (nuvem): treina nos 5172 e-mails e devolve a probabilidade de spam de uma base nova.

Roda no ambiente separado do cliente da nuvem (o tabpfn-client exige outras versões de pandas e sklearn):
    .venv-tabpfn-nuvem\\Scripts\\python.exe scripts\\tabpfn_nuvem_prever.py dados\\emails.csv dados\\spamassassin_vetores.npz saida.npz

Precisa de conta na Prior Labs, licença aceita e da chave na variável de ambiente TABPFN_TOKEN (nunca num arquivo).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TABPFN_NO_BROWSER", "1")
if not os.environ.get("TABPFN_TOKEN"):
    sys.exit("TABPFN_TOKEN ausente")
from tabpfn_client import TabPFNClassifier  # noqa: E402

csv, teste, saida = sys.argv[1:4]
df = pd.read_csv(csv); y = df.pop("Prediction").values; df.pop("Email No.")
X = df.values.astype(np.float32)
Z = np.load(teste); Xt = Z["X"].astype(np.float32)
for t in range(4):
    try:
        t0 = time.time(); m = TabPFNClassifier(); m.fit(X, y)
        p = m.predict_proba(Xt)[:, 1]
        break
    except Exception as e:                      # já houve queda de DNS no meio de uma chamada
        if t == 3:
            raise
        print(f"falha ({type(e).__name__}); nova tentativa em {30 * (t + 1)} s", flush=True); time.sleep(30 * (t + 1))
np.savez(saida, zte=p, segundos=time.time() - t0)
print(f"ok: {len(p)} previsões em {time.time() - t0:.0f} s -> {saida}")
