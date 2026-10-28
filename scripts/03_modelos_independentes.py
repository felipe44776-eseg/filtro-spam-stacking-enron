# %% [markdown]
# # 03. Torneio de 22 modelos independentes
#
# **Pergunta:** qual tipo de modelo lida melhor com estas contagens de palavras?
#
# **Protocolo.** Os mesmos 5 folds para todos (`StratifiedGroupKFold`, cópias idênticas sempre do mesmo lado).
# Rodada 1: os 22 modelos, 1 × 5 folds. Rodada 2: os 5 melhores em 3 × 5 folds, com teste t corrigido
# (Nadeau-Bengio) + Holm e o teste bayesiano correlacionado.
#
# Os resultados do experimento 01 são reaproveitados (`experimentos/01_benchmark`); se o arquivo não existir,
# o torneio roda do zero (cerca de 11 min) e retoma de onde parou se cair.

# %%
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from spamfilter import caminhos, dados, estatistica, graficos, metricas, modelos, saidas
from spamfilter.dados import SEED

base = dados.carregar()
X, y, g = base.X, base.y, base.g
CATALOGO = modelos.torneio()


def avaliar(nome, semente):
    linhas = []
    for f, (tr, te) in enumerate(dados.divisoes(y, g, semente)):
        m = CATALOGO[nome](); t0 = time.time(); m.fit(X[tr], y[tr]); t1 = time.time()
        yh = m.predict(X[te]); s = modelos.nota(m, X[te])
        linhas.append(dict(modelo=nome, seed=semente, fold=f, **metricas.metricas_simples(y[te], yh, s),
                           fit_s=t1 - t0, n_tr=len(tr), n_te=len(te)))
    return linhas


def rodar(arquivo, nomes, sementes, historico=None):
    """Salva a cada modelo e retoma de onde parou. Começa do resultado histórico, se houver."""
    arquivo = caminhos.RESULTADOS / arquivo
    if not arquivo.exists() and historico is not None and historico.exists():
        h = pd.read_csv(historico, sep=";", decimal=",").rename(columns={"fp_ham": "real_lixeira", "fn_spam": "spam_passou"})
        caminhos.RESULTADOS.mkdir(exist_ok=True); h.to_csv(arquivo, sep=";", decimal=",", index=False)
    feitos = pd.read_csv(arquivo, sep=";", decimal=",") if arquivo.exists() else pd.DataFrame(columns=["modelo", "seed"])
    for s in sementes:
        for nome in nomes:
            if ((feitos.modelo == nome) & (feitos.seed == s)).any():
                continue
            pd.DataFrame(avaliar(nome, s)).to_csv(arquivo, sep=";", decimal=",", index=False, mode="a", header=not arquivo.exists())
            print(f"semente {s} | {nome}", flush=True)
    return pd.read_csv(arquivo, sep=";", decimal=",")

# %% [markdown]
# ## Rodada 1: os 22 modelos

# %%
H = caminhos.EXPERIMENTOS / "01_benchmark"
R1 = rodar("03_torneio_rodada1.csv", list(CATALOGO), [SEED], H / "bench_rodada1.csv")
tab = (R1.groupby("modelo").agg(mcc=("mcc", "mean"), mcc_dp=("mcc", "std"), pr_auc=("pr_auc", "mean"),
                                real_lixeira=("real_lixeira", "mean"), spam_passou=("spam_passou", "mean"), treino_s=("fit_s", "mean"))
       .sort_values("mcc", ascending=False))
tab["familia"] = [modelos.FAMILIA_TORNEIO[n] for n in tab.index]
tab.index = [modelos.NOMES_TORNEIO.get(n, n) for n in tab.index]
mostra = tab.copy(); mostra[["real_lixeira", "spam_passou"]] *= 100
print(mostra.round({"mcc": 3, "mcc_dp": 3, "pr_auc": 3, "real_lixeira": 1, "spam_passou": 1, "treino_s": 2}).to_string())

fig, ax = plt.subplots(figsize=(7.4, 7))
t = tab.iloc[::-1]
ax.barh(range(len(t)), t.mcc.clip(lower=0), height=0.62, color=[graficos.FAMILIA_COR[f] for f in t.familia])
for i, v in enumerate(t.mcc):
    ax.text(max(v, 0) + 0.008, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8.5)
ax.set_yticks(range(len(t)), t.index, fontsize=9); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1.08)
ax.set_xlabel("MCC (média de 5 folds)"); ax.set_title("Torneio: 22 modelos, os mesmos 5 folds")
from matplotlib.patches import Patch
fams = [f for f in graficos.FAMILIA_COR if f in set(t.familia)]
ax.legend(handles=[Patch(color=graficos.FAMILIA_COR[f], label=f) for f in fams], loc="lower right", fontsize=8.5)
graficos.mostrar(fig, "03_torneio")

# %% [markdown]
# ## O detalhe que virou o jogo: TF-IDF com log
# Contar "spam" 10 vezes não é 10 vezes mais evidência que contar 1 vez. O log (`sublinear_tf`) achata as
# repetições, e os modelos que dependem de distância e de pesos lineares melhoram muito.

# %%
def mcc_cv(criar):
    return float(np.mean([metricas.mcc_np(y[te], criar().fit(X[tr], y[tr]).predict(X[te])) for tr, te in dados.divisoes(y, g, SEED)]))

lr = lambda tf: (lambda: make_pipeline(*tf, LogisticRegression(C=10, max_iter=3000, class_weight="balanced")))
knn = lambda tf: (lambda: make_pipeline(*tf, KNeighborsClassifier(5, metric="cosine")))
REPS = {"contagem crua": [], "TF-IDF": [TfidfTransformer()], "TF-IDF com log": [TfidfTransformer(sublinear_tf=True)]}
ACH = pd.DataFrame({rot: {"Reg. Logística": mcc_cv(lr(tf)), "kNN cosseno": mcc_cv(knn(tf))} for rot, tf in REPS.items()})
print(ACH.round(3).to_string())

# %% [markdown]
# ## Rodada 2: os 5 melhores em 3 × 5 folds

# %%
top = list(R1.groupby("modelo").mcc.mean().sort_values(ascending=False).index[:5])
R2 = rodar("03_torneio_rodada2.csv", top, [7, 2026], H / "bench_rodada2.csv")
RR = pd.concat([R1[R1.modelo.isin(top)], R2[R2.modelo.isin(top)]])
P = RR.pivot_table(index=["seed", "fold"], columns="modelo", values="mcc")
melhor = P.mean().idxmax()
C = estatistica.comparar(P, [(melhor, o) for o in P.columns.drop(melhor)], RR.n_tr.mean(), RR.n_te.mean(), runs=3)
print(f"melhor em 15 folds: {melhor} ({P[melhor].mean():.4f})")
print(C[["A", "B", "dif", "p", "p_holm", "P_A_melhor", "P_equivalentes", "P_B_melhor"]].round(4).to_string(index=False))

# %% [markdown]
# **Leitura.** Os modelos lineares sobre TF-IDF com log formam um pelotão (MCC ≈ 0,96); o boosting, que costuma
# vencer em tabelas, fica atrás. O stacking simples (4 modelos) lidera, mas **não** se separa estatisticamente do
# LinearSVC depois da correção de Holm: com 15 folds, diferenças abaixo de ~0,01 não são detectáveis.

# %%
saidas.gravar("03_torneio", {
    "torneio": [dict(modelo=n, **r) for n, r in tab.iterrows()],
    "achados": dict(tfidf_log=ACH.to_dict(), lr_mcc_antes=ACH.loc["Reg. Logística", "TF-IDF"], lr_mcc_depois=ACH.loc["Reg. Logística", "TF-IDF com log"],
                    knn_mcc_antes=ACH.loc["kNN cosseno", "TF-IDF"], knn_mcc_depois=ACH.loc["kNN cosseno", "TF-IDF com log"],
                    votacao_mcc=float(tab.loc["Votação (LR+CNB+RF+LGBM)", "mcc"]), lr_mcc=float(tab.loc["Reg. Logística TF-IDF", "mcc"])),
    "rodada2": dict(melhor=melhor, mcc_15_folds=P.mean().to_dict(), comparacoes=C),
}, fonte="scripts/03_modelos_independentes.py")
print("gravado: resultados/03_torneio.json")
