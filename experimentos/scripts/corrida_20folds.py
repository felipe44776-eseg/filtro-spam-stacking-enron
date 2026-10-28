"""Junta corrida + verificacao (20 folds, sementes 201-204): cada candidato contra o padrao."""
import sys, numpy as np, pandas as pd
from scipy import stats

rc = pd.read_csv(sys.argv[1], sep=";", decimal=",")
rho = (rc.n_te / rc.n_tr).mean()

def posterior(d):
    d = np.asarray(d); n = len(d); s = np.sqrt((1 / n + rho) * d.var(ddof=1))
    return d.mean(), float(1 - stats.t.cdf(0, n - 1, loc=d.mean(), scale=s))

print("20 folds (sementes 201-204) | dif positiva = candidato melhor que o padrão")
for met, sinal in [("mcc", 1), ("mcc_custo", 1), ("custo_por_mil", -1)]:
    P = rc.pivot(index="split", columns="candidato", values=met)
    lin = sorted(((c, *posterior(sinal * (P[c] - P["padrão"]))) for c in P.columns.drop("padrão")), key=lambda t: -t[2])
    print(f"\n{met} | padrão = {P['padrão'].mean():.4f}")
    for c, m, p in lin:
        print(f"  {c:18s} {P[c].mean():.4f}  dif {m:+.4f}  P(melhor que o padrão) {p:.2f}")
