# %% [markdown]
# # 07. Contra-estudo: tentando derrubar as próprias conclusões
#
# Seis perguntas que um avaliador cético faria:
# 1. **O MCC vem de sinal ou de acaso?** Com os rótulos embaralhados, o MCC deveria ir a zero (teste de permutação).
# 2. **Quase-cópias inflam o resultado?** O agrupamento só junta cópias (linhas de contagem iguais); e as quase iguais?
# 3. **O filtro depende do vocabulário da Enron?** Tirando as palavras da empresa, quanto cai? (ablação)
# 4. **Quantos e-mails são precisos?** Curva de aprendizado, com o TabPFN, que promete brilhar com poucos dados.
# 5. **A escolha depende do peso dado a cada erro?** Sensibilidade ao custo λ (1, 3, 9 e 20).
# 6. **Onde o filtro erra?** Os e-mails reais que foram para a lixeira, com o texto original.

# %%
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

from spamfilter import caminhos, dados, externo, graficos, metricas, modelos, representacao, saidas
from spamfilter.dados import SEED
from spamfilter.motor import Motor

base = dados.carregar()
X, y, g = base.X, base.y, base.g
vocab = np.array(base.vocab)
lr = lambda: make_pipeline(representacao.tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced"))
DIV = dados.divisoes(y, g, SEED)

# %% [markdown]
# ## 1. Teste de sinal (permutação)
# Reg. Logística com TF-IDF log, nas 5 divisões da semente 42, com os rótulos verdadeiros e com 999 embaralhamentos.
# Com N permutações, o menor p possível é 1/(N+1). **O que o teste mostra:** o MCC não vem do acaso. **O que ele
# não mostra:** ausência de vazamento (cópias que cruzassem a divisão levariam o rótulo embaralhado junto). A
# proteção contra vazamento é o agrupamento das cópias (conferido em `tests/test_dados.py`) e a seção 2.

# %%
def mcc_cv(yy, Xm=X):
    return float(np.mean([metricas.mcc_np(yy[te], lr().fit(Xm[tr], yy[tr]).predict(Xm[te])) for tr, te in DIV]))

N_PERM = 999
mcc_real = mcc_cv(y)
rng = np.random.default_rng(SEED)
perms = [rng.permutation(y) for _ in range(N_PERM)]
mcc_perm = Parallel(n_jobs=4)(delayed(mcc_cv)(p) for p in perms)
p_valor = (1 + sum(v >= mcc_real for v in mcc_perm)) / (1 + len(mcc_perm))
print(f"MCC com os rótulos verdadeiros: {mcc_real:.3f} | com rótulos embaralhados ({N_PERM} vezes): média {np.mean(mcc_perm):+.3f}, "
      f"máximo {np.max(mcc_perm):+.3f} | p = {p_valor:.3f} (o menor possível com {N_PERM} permutações é {1 / (N_PERM + 1):.3f})")

# %% [markdown]
# ## 2. Quase-cópias
# O protocolo põe as cópias (linhas de contagem **iguais**) do mesmo lado da divisão. O mesmo aviso com outro dia
# já é cópia, porque algarismos não são contados. Quase-cópias (o mesmo aviso com uma ou outra palavra diferente, o
# mesmo spam com outro texto no link) podem cair dos dois lados. Medida: para cada e-mail de teste, o cosseno com o
# e-mail de treino mais parecido (TF-IDF log). Depois, o MCC do Stacking 7 padrão com e sem os e-mails de teste
# que têm um vizinho acima de 0,95 no treino, nos 20 folds de confirmação. Nenhum modelo é escolhido aqui.

# %%
from sklearn.preprocessing import normalize
V = normalize(representacao.tfidf().fit_transform(X))
def max_cos(a, b, lote=500):
    return np.concatenate([(V[a[i:i + lote]] @ V[b].T).toarray().max(1) for i in range(0, len(a), lote)])

mq = Motor(base, log=lambda *a: None)
C20q = mq.adicionar_divisoes([201, 202, 203, 204])
dev_t, teste_t = dados.indices_teste(base)
s_tt = max_cos(teste_t, dev_t)
qc = []
for c in C20q:
    tr = mq.dev[mq.splits[c]["tr"]]; tt = mq.dev[mq.splits[c]["tes"]["te"]]
    sc = max_cos(tt, tr); yy = y[tt]
    _, h, hc = mq.prever_stacking(modelos.STACKING7_PADRAO, c)
    sem = sc < 0.95
    qc.append(dict(fold=c, fracao_quase_copia=float(np.mean(~sem)), mcc=metricas.mcc_np(yy, h), mcc_sem_quase_copias=metricas.mcc_np(yy[sem], h[sem]),
                   mcc_custo=metricas.mcc_np(yy, hc), mcc_custo_sem_quase_copias=metricas.mcc_np(yy[sem], hc[sem])))
QC = pd.DataFrame(qc)
QUASE = dict(limiar_cosseno=0.95, teste_trancado_fracao=float(np.mean(s_tt >= 0.95)),
             teste_trancado_fracao_email_real=float(np.mean(s_tt[y[teste_t] == 0] >= 0.95)),
             teste_trancado_fracao_spam=float(np.mean(s_tt[y[teste_t] == 1] >= 0.95)),
             folds_fracao=float(QC.fracao_quase_copia.mean()), **{k: float(QC[k].mean()) for k in
             ["mcc", "mcc_sem_quase_copias", "mcc_custo", "mcc_custo_sem_quase_copias"]})
print(f"teste trancado: {QUASE['teste_trancado_fracao']:.1%} dos e-mails têm quase-cópia (cosseno ≥ 0,95) no desenvolvimento "
      f"({QUASE['teste_trancado_fracao_email_real']:.1%} dos e-mails reais, {QUASE['teste_trancado_fracao_spam']:.1%} dos spams)")
print(f"20 folds: {QUASE['folds_fracao']:.1%} dos e-mails de teste têm quase-cópia no treino | Stacking 7 padrão: MCC "
      f"{QUASE['mcc']:.4f} → {QUASE['mcc_sem_quase_copias']:.4f} sem elas (limiar de custo: {QUASE['mcc_custo']:.4f} → "
      f"{QUASE['mcc_custo_sem_quase_copias']:.4f})")

# %% [markdown]
# ## 3. Ablação: sem o vocabulário da Enron
# Palavras da empresa e de pessoas (ex.: "enron", "hpl", "daren", "ect", "hou") e o artefato da coleta de spam
# ("paliourg", de Paliouras, um dos pesquisadores que montaram a base).

# %%
DOMINIO = ["enron", "daren", "farmer", "hpl", "sitara", "mmbtu", "ect", "hou", "melissa", "cotten", "chokshi", "tenaska",
           "forwarded", "paliourg", "clynes", "pat", "ami", "gary", "robert", "smith", "george", "mary", "noms", "nom",
           "actuals", "xls", "meter", "teco", "volumes", "nomination", "btu", "aimee", "lauri", "stacey", "susan"]
dom = np.array([w in DOMINIO for w in vocab]); curta = np.array([len(w) < 3 for w in vocab])
Xext, yext, _ = externo.carregar_externo(base)
Xsa, ysa, _, _ = externo.carregar_spamassassin(base)
Xe = __import__("scipy").sparse.vstack(list(Xext.values())); ye = np.concatenate(list(yext.values()))

def avalia_mascara(manter):
    dentro = mcc_cv(y, X[:, manter])
    mfin = lr().fit(X[:, manter], y)
    fora = {nome: dict(mcc=metricas.mcc_np(yy, mfin.predict(XX[:, manter])), pr_auc=average_precision_score(yy, mfin.predict_proba(XX[:, manter])[:, 1]))
            for nome, XX, yy in [("Enron2-6", Xe, ye), ("SpamAssassin", Xsa, ysa)]}
    return dict(mcc_esta_caixa=dentro, mcc_enron2_6=fora["Enron2-6"]["mcc"], pr_auc_enron2_6=fora["Enron2-6"]["pr_auc"],
                mcc_spamassassin=fora["SpamAssassin"]["mcc"], pr_auc_spamassassin=fora["SpamAssassin"]["pr_auc"])

ABL = pd.DataFrame({"todas as 3000 colunas": avalia_mascara(np.ones(len(vocab), bool)),
                    f"sem as {int(dom.sum())} palavras da Enron": avalia_mascara(~dom),
                    "sem as da Enron e sem 1-2 letras": avalia_mascara(~dom & ~curta)}).T
print(f"colunas da Enron encontradas no vocabulário: {int(dom.sum())}")
print(ABL.round(3).to_string())

# %% [markdown]
# ## 4. Curva de aprendizado (5 divisões da semente 201; o teste é sempre o mesmo)

# %%
m = Motor(base, cache=caminhos.RESULTADOS / "cache_motor", leitura_extra=[caminhos.CACHE_MOTOR], log=print)
CONF = m.adicionar_divisoes([201])
TAM = [250, 500, 1000, 2000]
_, ydev, _ = m.bases["dev"]
for f, c in enumerate(CONF):
    tr, te = m.splits[c]["tr"], m.splits[c]["tes"]["te"]
    r = np.random.default_rng(1000 + f)
    for n in TAM:
        sub = np.concatenate([r.choice(tr[ydev[tr] == k], int(round(n * (ydev[tr] == k).mean())), replace=False) for k in (0, 1)])
        m.splits[f"a{n}_{f}"] = dict(base="dev", tr=np.sort(sub), tes={"te": te}, si=201 * 100 + f)
CURVA = [f"a{n}_{f}" for n in TAM for f in range(5)]
P7, LS = modelos.STACKING7_PADRAO, modelos.LINEARSVC_PADRAO
m.rodar(m.tarefas_stacking(CURVA, [P7], linearsvc=[LS]), "curva", blocos=4)

# TabPFN na nuvem (se houver chave; senão usa o arquivo já gravado)
arq_tab = caminhos.RESULTADOS / "cache_tabpfn" / "curva.npz"
ent = caminhos.RESULTADOS / "cache_tabpfn" / "curva_entrada.npz"
py_nuvem = caminhos.RAIZ / ".venv-tabpfn-nuvem" / "Scripts" / "python.exe"
if not arq_tab.exists() and os.environ.get("TABPFN_TOKEN") and py_nuvem.exists():
    ent.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ent, chaves=np.array(CURVA), **{f"tr_{k}": m.dev[m.splits[k]["tr"]] for k in CURVA}, **{f"te_{k}": m.dev[m.splits[k]["tes"]["te"]] for k in CURVA})
    subprocess.run([str(py_nuvem), str(caminhos.RAIZ / "scripts" / "tabpfn_nuvem_lotes.py"), str(caminhos.CSV), str(ent), str(arq_tab)], check=True)
TAB = dict(np.load(arq_tab)) if arq_tab.exists() else {}
T20 = pd.read_csv(caminhos.EXPERIMENTOS / "05_tabpfn" / "tabpfn_20folds.csv", sep=";", decimal=",")

linhas = []
for f, c in enumerate(CONF):
    for n, chave in [(n, f"a{n}_{f}") for n in TAM] + [(len(m.splits[c]["tr"]), c)]:
        yte = m.y_teste(chave)
        linhas.append(dict(n=n, fold=f, metodo="Stacking 7 padrão", mcc=metricas.mcc_np(yte, m.prever_stacking(P7, chave)[1])))
        linhas.append(dict(n=n, fold=f, metodo="LinearSVC", mcc=metricas.mcc_np(yte, m.prever_linearsvc(LS, chave)[1])))
        for fam, rot in [("Reg. Logística", "Reg. Logística"), ("kNN cosseno", "kNN cosseno")]:
            linhas.append(dict(n=n, fold=f, metodo=rot, mcc=metricas.mcc_np(yte, m.carregar(chave, fam, 0)[1] > 0.5)))
        if f"zte_{chave}" in TAB:
            linhas.append(dict(n=n, fold=f, metodo="TabPFN-3.5-Plus", mcc=metricas.mcc_np(yte, TAB[f"zte_{chave}"] > 0.5)))
    tfull = T20[(T20.metodo == "TabPFN-3.5-Plus") & (T20.split == c)]
    if len(tfull):
        linhas.append(dict(n=len(m.splits[c]["tr"]), fold=f, metodo="TabPFN-3.5-Plus", mcc=float(tfull.mcc.iloc[0])))
CUR = pd.DataFrame(linhas)
CUR["n"] = CUR.n.where(CUR.n.isin(TAM), 3310)          # o treino completo tem 3309 ou 3310 e-mails conforme o fold
curva = CUR.groupby(["metodo", "n"]).mcc.agg(["mean", "std"]).reset_index()
print(curva.pivot(index="n", columns="metodo", values="mean").round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 3.8))
cores = {"Stacking 7 padrão": graficos.CATEGORICA[0], "LinearSVC": graficos.TINTA_MUDA, "Reg. Logística": graficos.CATEGORICA[1],
         "kNN cosseno": graficos.CATEGORICA[2], "TabPFN-3.5-Plus": graficos.CATEGORICA[6]}
for mt, d in curva.groupby("metodo"):
    ax.plot(d.n, d["mean"], marker="o", ms=6, color=cores[mt], mec=graficos.SUPERFICIE, mew=1.5)
    ax.text(d.n.iloc[-1] * 1.06, d["mean"].iloc[-1], mt, va="center", fontsize=8.5, color=graficos.TINTA)
ax.set_xscale("log"); ax.set_xticks([250, 500, 1000, 2000, 3300], ["250", "500", "1000", "2000", "≈3300"])
ax.set_xlim(200, 9000); ax.set_xlabel("e-mails de treino"); ax.set_ylabel("MCC")
ax.set_title("Com poucos e-mails, quem aprende mais rápido?")
graficos.mostrar(fig, "07_curva_aprendizado")

# %% [markdown]
# ## 5. Sensibilidade ao custo: o vencedor muda com o peso do erro?
# λ = quantos spams recebidos equivalem a 1 e-mail real perdido. O limiar é re-escolhido para cada λ (só no treino).
# Entram também o Stacking 8 e o TabPFN sozinho (notas da nuvem já gravadas) e a referência "não filtrar".

# %%
mc = Motor(base, log=lambda *a: None)
C20 = mc.adicionar_divisoes([201, 202, 203, 204]); mc.adicionar_externo(Xext)
TABN = caminhos.EXPERIMENTOS / "05_tabpfn" / "cache_tabpfn" / "nuvem"
def tab(chave, t):
    z = np.load(TABN / f"{chave}.npz"); return z["ztr"], z[f"zte_{t}"]
def prever_lam(tipo, cfg, c, t, lam):
    if tipo == "s":
        return mc.prever_stacking(cfg, c, t, custo_fp=lam)
    if tipo == "s8":
        return mc.prever_stacking(cfg, c, t, extra=tab(c, t), custo_fp=lam)
    if tipo == "l":
        return mc.prever_linearsvc(cfg, c, t, custo_fp=lam)
    if tipo == "tab":
        ztr, zte = tab(c, t); lim = metricas.limiar_custo(ztr, mc.y_treino(c), lam, 1)
        return zte, (zte > 0.5).astype(int), (zte > lim).astype(int)
    n = len(yext[t]) if c == "externo" else len(mc.y_teste(c))           # não filtrar
    return np.zeros(n), np.zeros(n, int), np.zeros(n, int)
CAND = {"Stacking 7 padrão": ("s", modelos.STACKING7_PADRAO), "Stacking 7 especialista": ("s", modelos.STACKING7_ESPECIALISTA),
        "Stacking 8 padrão (+TabPFN)": ("s8", modelos.STACKING7_PADRAO), "TabPFN-3.5-Plus sozinho": ("tab", None),
        "LinearSVC padrão": ("l", modelos.LINEARSVC_PADRAO), "LinearSVC otimizado": ("l", modelos.LINEARSVC_OTIMIZADO),
        "não filtrar": ("nada", None)}
lin = []
for lam in [1, 3, 9, 20]:
    for nome, (tipo, cfg) in CAND.items():
        for onde, chaves, te in [("esta caixa (20 folds)", C20, None), ("Enron2-6", ["externo"], list(Xext))]:
            for c in chaves:
                for t in (te or ["te"]):
                    yv = yext[t] if onde == "Enron2-6" else mc.y_teste(c)
                    r = prever_lam(tipo, cfg, c, t, lam)
                    lin.append(dict(custo_fp=lam, onde=onde, metodo=nome, custo_por_mil=metricas.custo_por_mil(yv, r[2], lam, 1),
                                    real_lixeira=(r[2][yv == 0] == 1).mean(), spam_passou=(r[2][yv == 1] == 0).mean()))
SENS = pd.DataFrame(lin).groupby(["onde", "custo_fp", "metodo"]).mean().reset_index()
venc = SENS.loc[SENS.groupby(["onde", "custo_fp"]).custo_por_mil.idxmin(), ["onde", "custo_fp", "metodo", "custo_por_mil"]]
print(SENS.pivot_table(index=["onde", "metodo"], columns="custo_fp", values="custo_por_mil").round(1).to_string())
print("\nmais barato para cada λ:\n" + venc.round(1).to_string(index=False))

# %% [markdown]
# ## 6. Onde o filtro erra: e-mails reais que foram para a lixeira (Especialista, 20 folds)

# %%
textos, _, _ = externo.ler_tar(1)
X1 = externo.vetorizar(textos, vocab)
por_linha = {}
for t, v in zip(textos, X1):
    por_linha.setdefault(v.astype(np.int64).tobytes(), t)
A = X.toarray().astype(np.int64)
exemplos = []
for c in C20:
    te = mc.splits[c]["tes"]["te"]; yte = mc.y_teste(c)
    p, _, yc = mc.prever_stacking(modelos.STACKING7_ESPECIALISTA, c)
    for j in np.flatnonzero((yc == 1) & (yte == 0)):
        idx = int(mc.dev[te[j]])
        exemplos.append(dict(divisao=c, linha_csv=idx, prob_spam=float(p[j]), texto=por_linha[A[idx].tobytes()][:400]))
EX = pd.DataFrame(exemplos).drop_duplicates("linha_csv").sort_values("prob_spam", ascending=False)
print(f"e-mails reais distintos na lixeira em 20 folds: {len(EX)} (de {int((mc.bases['dev'][1] == 0).sum())} e-mails reais do desenvolvimento)")
for _, r in EX.head(5).iterrows():
    print(f"\n--- prob. de spam {r.prob_spam:.3f} ---\n{r.texto[:300]}")

# palavras com mais peso na Reg. Logística (treino nos 5172)
mfin = lr().fit(X, y); w = mfin[-1].coef_.ravel()
longa = np.array([len(v) >= 3 for v in vocab])
PES = pd.DataFrame({"palavra": vocab, "peso": w})[longa]
pes_spam, pes_real = PES.nlargest(15, "peso"), PES.nsmallest(15, "peso")
print("\nempurram para spam:", ", ".join(pes_spam.palavra)); print("empurram para e-mail real:", ", ".join(pes_real.palavra))

# %%
saidas.gravar("07_contra_estudo", {
    "contra_estudo": dict(
        permutacao=dict(mcc_real=mcc_real, mcc_permutado=mcc_perm, p=p_valor, n_permutacoes=N_PERM, modelo="Reg. Logística TF-IDF log (C=10)",
                        mcc_permutado_media=float(np.mean(mcc_perm)), mcc_permutado_max=float(np.max(mcc_perm))),
        quase_copias=QUASE,
        ablacao=ABL.reset_index().rename(columns={"index": "variante"}), palavras_enron=[w for w in DOMINIO if w in set(vocab)],
        curva_aprendizado=curva, sensibilidade_custo=SENS, vencedor_por_custo=venc,
        emails_reais_na_lixeira=EX.head(10), n_emails_reais_na_lixeira=len(EX),
        pesos_palavras=dict(spam=pes_spam, email_real=pes_real)),
}, fonte="scripts/07_contra_estudo.py")
print("gravado: resultados/07_contra_estudo.json")
