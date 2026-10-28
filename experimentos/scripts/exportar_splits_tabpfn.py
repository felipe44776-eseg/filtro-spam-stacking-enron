"""Exporta as divisoes do motor (exp. 03/04) como indices, para o TabPFN usar exatamente os mesmos e-mails
em qualquer ambiente (a .venv-tabpfn-nuvem tem outras versoes de pandas/sklearn)."""
import hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]; E = RAIZ / "experimentos"
df = pd.read_csv(RAIZ / "dados" / "emails.csv"); y_all = df.pop("Prediction").values; df.pop("Email No.")
g_all = pd.util.hash_pandas_object(df, index=False).values          # igual ao motor
A = np.ascontiguousarray(df.values, dtype=np.int64); hash_csv = {hashlib.md5(r.tobytes()).digest() for r in A}
ix = np.load(E / "indices_teste.npz"); dev, teste = ix["dev"], ix["teste"]
y, g = y_all[dev], g_all[dev]
out, chaves = {"dev": dev, "teste": teste}, []

def add(chave, tr_glob, te_glob, ytr, gtr, si):
    out[f"{chave}__tr"] = tr_glob
    if te_glob is not None:
        out[f"{chave}__te"] = te_glob
    for k, (itr, ite) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=si).split(np.zeros(len(ytr)), ytr, gtr)):
        out[f"{chave}__in{k}_tr"], out[f"{chave}__in{k}_te"] = itr, ite      # relativos ao treino, como no motor
    chaves.append(chave)

for s in [201, 202, 203, 204]:                                       # divisoes de confirmacao (exp. 03) = corrida + verificacao (exp. 04)
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y, g)):
        add(f"c{s}_{f}", dev[tr], dev[te], y[tr], g[tr], s * 100 + f)
add("externo", np.arange(len(y_all)), None, y_all, g_all, 42)       # treino nos 5172 para a base externa

z = np.load(RAIZ / "dados" / "enron_vetores.npz")
for k in range(2, 7):
    novo = np.array([hashlib.md5(r.tobytes()).digest() not in hash_csv for r in z[f"X{k}"].astype(np.int64)])
    out[f"ext_Enron{k}"] = np.flatnonzero(novo)
out["chaves"] = np.array(chaves)
destino = E / "05_tabpfn" / "splits_tabpfn.npz"
np.savez(destino, **out)
print(f"{len(chaves)} divisões exportadas -> {destino}")
print("conferência c201_0:", len(out["c201_0__tr"]), "treino /", len(out["c201_0__te"]), "teste; interno 0:",
      len(out["c201_0__in0_tr"]), "/", len(out["c201_0__in0_te"]))
print("externo:", {k: len(out[f"ext_Enron{k}"]) for k in range(2, 7)})
