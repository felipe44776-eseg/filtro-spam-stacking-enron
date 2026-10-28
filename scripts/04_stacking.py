# %% [markdown]
# # 04. Ninguém vence sozinho: o stacking
#
# **Ideia.** Em vez de escolher um modelo, treinamos vários de famílias diferentes e um **meta-modelo**
# (Regressão Logística) aprende quanto confiar em cada um. Funciona se os modelos **erram em e-mails diferentes**.
#
# **Protocolo (experimento 02).** Base de teste trancada com 20%; nos 80% restantes, validação cruzada 3 × 5 por
# grupo; dentro de cada treino, outra CV por grupo gera as notas *out-of-fold* que alimentam o meta-modelo (assim
# ele nunca vê a nota de um modelo sobre um e-mail que esse modelo usou para treinar).
#
# **Stacking 7:** um representante por família: linear (Reg. Logística), kernel (SVM RBF), vizinhos (kNN),
# rede neural (MLP), probabilística (Naive Bayes), *bagging* (Extra Trees) e *boosting* (LightGBM).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spamfilter import caminhos, estatistica, graficos, metricas, modelos, saidas
from spamfilter.dados import carregar
from spamfilter.motor import Motor

H = caminhos.EXPERIMENTOS / "02_stacking"
R = pd.read_csv(H / "stacking_cv.csv", sep=";", decimal=",")
resumo = (R.groupby(["metodo", "tipo"]).agg(mcc=("mcc", "mean"), mcc_dp=("mcc", "std"), pr_auc=("pr_auc", "mean"),
                                            real_lixeira=("real_lixeira", "mean"), spam_passou=("spam_passou", "mean"), treino_s=("treino_s", "mean"))
          .sort_values("mcc", ascending=False).reset_index(level="tipo"))
mostra = resumo.copy(); mostra[["real_lixeira", "spam_passou"]] *= 100
print(mostra.round(4).to_string())

# %% [markdown]
# ## 1. Mais modelos não é melhor
# Com 13 modelos, o stacking empata com o de 7; a seleção gulosa de Caruana e a votação simples ficam atrás.
# O que importa é a **diversidade**, não a quantidade.

# %%
P15 = R.pivot_table(index=["seed", "fold"], columns="metodo", values="mcc")
pares = [("Stacking 7 (1 por família)", o) for o in ["Stacking 13 (todos)", "Stacking 13 + L1", "Caruana 13", "Média 13 (votação)",
                                                     "Stacking 4 (LR+NB+RF+LGBM)", "LinearSVC"]]
C15 = estatistica.comparar(P15, pares, R.n_tr.mean(), R.n_te.mean(), runs=3)
print(C15[["A", "B", "dif", "p_holm", "P_A_melhor", "P_equivalentes", "P_B_melhor"]].round(3).to_string(index=False))

# %% [markdown]
# ## 2. Quem erra o quê (20 folds de confirmação, experimento 03)
# Três medidas, para cada membro:
# - **P(erra | outro erra):** quando outro membro erra um e-mail, com que frequência este erra junto. Baixo = traz
#   informação que os outros não têm.
# - **Acerta quando a maioria erra:** quantos e-mails ele salva quando 4 ou mais dos 7 erram.
# - **Jaccard** (dos e-mails que pelo menos um dos dois errou, a fração que os dois erraram): é simétrico, mas
#   favorece quem erra muito, porque a união cresce com o número de erros. Fica só como referência.

# %%
base = carregar()
m = Motor(base, log=lambda *a: None)
CONF = m.adicionar_divisoes([201, 202, 203, 204])
P = modelos.STACKING7_PADRAO
LIMIAR0 = {f: (0.0 if f == "SVM RBF" else 0.5) for f in modelos.FAMILIAS}      # SVC sem probabilidade: distância à fronteira
erros, ys = {f: [] for f in modelos.FAMILIAS}, []
for c in CONF:
    yte = m.y_teste(c); ys.append(yte)
    for i, f in enumerate(modelos.FAMILIAS):
        erros[f].append((m.carregar(c, f, P[i])[1] > LIMIAR0[f]).astype(int) != yte)
    _, yh, _ = m.prever_stacking(P, c); erros.setdefault("Stacking 7", []).append(yh != np.concatenate([yte])[: len(yh)])
E = pd.DataFrame({f: np.concatenate(v) for f, v in erros.items()})
fams = modelos.FAMILIAS
J = pd.DataFrame([[(E[a] & E[b]).sum() / max((E[a] | E[b]).sum(), 1) for b in fams] for a in fams], index=fams, columns=fams)
so_ele = pd.Series({f: int((E[f] & ~E[[o for o in fams if o != f]].any(axis=1)).sum()) for f in fams})
maioria_erra = E[fams].sum(axis=1) >= 4
resgate = pd.Series({f: int((~E[f] & maioria_erra).sum()) for f in fams})
# P(linha erra | coluna erra): assimétrica; a diagonal (1) fica de fora
PC = pd.DataFrame([[(E[a] & E[b]).sum() / max(E[b].sum(), 1) for b in fams] for a in fams], index=fams, columns=fams)
erra_junto = pd.Series({a: float(np.mean([PC.loc[a, b] for b in fams if b != a])) for a in fams})
tab_erros = pd.DataFrame({"erros": E[fams].sum(), "só ele erra": so_ele, "acerta quando a maioria (≥4) erra": resgate,
                          "P(erra | outro erra)": erra_junto.round(3), "Jaccard médio": J.where(~np.eye(len(fams), dtype=bool)).mean().round(3)})
print(f"e-mails de teste (20 folds): {len(E)} | erros do Stacking 7: {int(E['Stacking 7'].sum())} | "
      f"e-mails em que a maioria dos 7 erra: {int(maioria_erra.sum())}")
print(tab_erros.sort_values("P(erra | outro erra)").to_string())

fig, ax = plt.subplots(figsize=(6.4, 5.2))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", graficos.SEQUENCIAL)
Pm = np.ma.masked_where(np.eye(len(fams), dtype=bool), PC.values)
im = ax.imshow(Pm, cmap=cmap, vmin=0, vmax=1)
for a in range(len(fams)):
    for b in range(len(fams)):
        if a != b:
            v = PC.values[a, b]
            ax.text(b, a, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else graficos.TINTA)
ax.set_xticks(range(len(fams)), fams, rotation=35, ha="right"); ax.set_yticks(range(len(fams)), fams)
ax.set_xlabel("quando este erra..."); ax.set_ylabel("...com que frequência este erra junto")
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Erros em comum: quanto menor, mais informação nova o modelo traz")
fig.colorbar(im, ax=ax, shrink=0.7)
graficos.mostrar(fig, "04_erros_em_comum")

# %% [markdown]
# ## 3. Quanto o meta-modelo confia em cada um
# Pesos da Regressão Logística do meta-modelo do Stacking 7 padrão sobre as notas padronizadas: média e desvio
# padrão entre os 20 folds (os filtros entregues, treinados nos 5172 e-mails, têm pesos próprios: script 09).

# %%
W = pd.DataFrame([m.pesos_meta(P, c) for c in CONF])
pesos = W[fams].mean().sort_values()
pesos_dp = W[fams].std(ddof=1)[pesos.index]
print(pd.DataFrame({"média": pesos, "desvio padrão": pesos_dp}).round(2).to_string())
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(range(len(pesos)), pesos.values, height=0.55, xerr=pesos_dp.values, ecolor=graficos.TINTA_2, capsize=3,
        color=[graficos.FAMILIA_COR[graficos.FAMILIA_DO_MEMBRO[f]] for f in pesos.index])
for i, v in enumerate(pesos.values):
    ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:.2f}".replace(".", ","), va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_yticks(range(len(pesos)), pesos.index); ax.axvline(0, color=graficos.EIXO, lw=1); ax.grid(axis="y", visible=False)
ax.set_xlabel("peso no meta-modelo (nota padronizada)"); ax.set_title("O kNN, fraco sozinho, é o reforço mais valioso")
graficos.mostrar(fig, "04_pesos_meta")

# %% [markdown]
# ## 4. Stacking 7 × melhor modelo único (20 folds de confirmação)

# %%
linhas = []
for c in CONF:
    yte = m.y_teste(c)
    linhas.append(dict(split=c, metodo="Stacking 7", **metricas.metricas(yte, *m.prever_stacking(P, c))))
    linhas.append(dict(split=c, metodo="LinearSVC", **metricas.metricas(yte, *m.prever_linearsvc(modelos.LINEARSVC_PADRAO, c))))
    for i, f in enumerate(fams):
        s = m.carregar(c, f, P[i])[1]; yh = (s > LIMIAR0[f]).astype(int)
        linhas.append(dict(split=c, metodo=f, **metricas.metricas(yte, s, yh, yh)))
R20 = pd.DataFrame(linhas)
por_metodo = R20.groupby("metodo")[["mcc", "pr_auc", "real_lixeira", "spam_passou"]].mean().sort_values("mcc", ascending=False)
print(por_metodo.round(4).to_string())
P20 = R20.pivot(index="split", columns="metodo", values="mcc")
C20 = estatistica.comparar(P20, [("Stacking 7", "LinearSVC")], 3310, 828, runs=4)
print(C20[["A", "B", "dif", "p", "P_A_melhor", "P_equivalentes", "P_B_melhor"]].round(4).to_string(index=False))

# %% [markdown]
# **Leitura.** Os membros que mais salvam e-mails quando a maioria erra (as árvores e o kNN) erram menos junto com
# os outros e recebem pesos altos. O kNN, dos mais fracos sozinho, tem o maior peso médio. O Naive Bayes erra
# muito e erra junto com os outros: o meta-modelo lhe dá peso quase zero. É a definição prática de diversidade:
# o que vale é acertar onde os outros erram. Por isso o Stacking 7: modelos distintos que erram de formas distintas.

# %%
print("menos erra junto (P(erra | outro erra)):", ", ".join(f"{k} {v:.2f}" for k, v in erra_junto.sort_values().items()))
print("mais salva quando a maioria erra:", ", ".join(f"{k} {v}" for k, v in resgate.sort_values(ascending=False).items()))

# %%
ext = pd.read_csv(H / "stacking_externo.csv", sep=";", decimal=",").groupby("modelo")[["mcc", "pr_auc", "real_lixeira", "spam_passou"]].mean()
saidas.gravar("04_stacking", {
    "stacking": dict(variantes=resumo.reset_index().rename(columns={"metodo": "nome"}), comparacoes_15_folds=C15,
                     erros_em_comum=dict(modelos=fams, matriz=J.values.round(4), p_erra_se_outro_erra=PC.values.round(4),
                                         erra_junto=erra_junto.round(4).to_dict()),
                     tabela_erros=tab_erros.reset_index().rename(columns={"index": "familia"}),
                     pesos_meta=pesos.to_dict(), pesos_meta_dp=pesos_dp.to_dict(), confirmacao_20_folds=por_metodo.reset_index(), comparacao=C20,
                     externo_exp02=ext.reset_index()),
}, fonte="scripts/04_stacking.py")
print("gravado: resultados/04_stacking.json")
