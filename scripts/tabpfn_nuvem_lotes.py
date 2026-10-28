"""TabPFN-3.5-Plus (nuvem) em vários pares treino/teste de uma vez (curva de aprendizado).

Entrada: um .npz com `chaves` e, para cada chave k, `tr_k` e `te_k` (índices de linhas do CSV).
Saída: um .npz com `zte_k` (probabilidade de spam no teste). Retoma do que já existe na saída.
    .venv-tabpfn-nuvem\\Scripts\\python.exe scripts\\tabpfn_nuvem_lotes.py dados\\emails.csv entrada.npz saida.npz
Precisa da chave na variável de ambiente TABPFN_TOKEN.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

os.environ.setdefault("TABPFN_NO_BROWSER", "1")
if not os.environ.get("TABPFN_TOKEN"):
    sys.exit("TABPFN_TOKEN ausente")
from tabpfn_client import TabPFNClassifier  # noqa: E402

csv, entrada, saida = sys.argv[1:4]
df = pd.read_csv(csv); y = df.pop("Prediction").values; df.pop("Email No.")
X = df.values.astype(np.float32)
E = np.load(entrada); chaves = [str(c) for c in E["chaves"]]
feito = dict(np.load(saida)) if os.path.exists(saida) else {}


def um(k):
    if f"zte_{k}" in feito:
        return k, feito[f"zte_{k}"]
    tr, te = E[f"tr_{k}"], E[f"te_{k}"]
    for t in range(4):
        try:
            m = TabPFNClassifier(); m.fit(X[tr], y[tr]); return k, m.predict_proba(X[te])[:, 1]
        except Exception as e:
            if t == 3:
                raise
            print(f"{k}: falha ({type(e).__name__}); nova tentativa", flush=True); time.sleep(30 * (t + 1))


t0 = time.time()
with ThreadPoolExecutor(3) as ex:
    for k, z in ex.map(um, chaves):
        feito[f"zte_{k}"] = z
        np.savez(saida, **feito)
        print(f"ok {k} ({time.time() - t0:.0f} s)", flush=True)
