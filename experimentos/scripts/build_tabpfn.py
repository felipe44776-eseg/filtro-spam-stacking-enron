"""Experimento 05: TabPFN sozinho e como 8a familia do stacking. Reaproveita o motor do exp. 03 (cache dos 7 modelos)
e as notas do TabPFN geradas por tabpfn_notas.py (nuvem: 20 divisoes + externo; local: semente 201)."""
import nbformat as nbf, sys, runpy
from pathlib import Path

ga = runpy.run_path(str(Path(__file__).with_name("build_ga.py")), run_name="build_ga")
C0 = ga["C"]
base = C0[1:8]            # imports, dados, espaco de busca, motor
conf = [C0[14]]           # confirmacao: METODOS, limiar de custo, prever, metricas, divisoes c201-c204
trocas = [('"cache_ga").resolve()', '"../03_ga/cache_ga").resolve()'),
          ('"progresso_ga.log")', '"progresso_analise.log")'),
          ('; R.to_csv("ga_confirmacao.csv", sep=";", decimal=",", index=False)', '')]
for c in base + conf:
    for a, b in trocas:
        c.source = c.source.replace(a, b)
assert any("../03_ga/cache_ga" in c.source for c in base)

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Experimento 05: TabPFN, um modelo de fundação para tabelas, sozinho e dentro do stacking

**O que é o TabPFN.** É uma rede *transformer* pré-treinada pela Prior Labs em milhões de tabelas sintéticas. Ela não "treina" no sentido usual: recebe os e-mails de treino como contexto e prevê os novos numa única passada (*in-context learning*). Versões usadas:
- **TabPFN-3.5-Plus** na nuvem (`tabpfn-client`), nas 20 divisões e na base externa;
- **TabPFN-3.5-Fast** local, em CPU, nas 5 divisões da semente 201, como réplica.

**Perguntas**
1. Sozinho, o TabPFN chega perto do Stacking 7?
2. Como 8ª família, ele melhora o stacking? Ou seja, erra em e-mails diferentes dos outros 7?
3. O resultado se mantém nas outras caixas de e-mail (Enron2–6)?

**Protocolo.** As mesmas 20 divisões dos experimentos 03 e 04 (sementes 201 a 204), os mesmos treinos internos e o mesmo limiar de custo (1 e-mail real perdido = 9 spams). As notas dos 7 modelos vêm do cache do exp. 03. As do TabPFN foram geradas por `scripts/tabpfn_notas.py`, com as divisões exportadas por índice.

**Termos.** *E-mail real* = e-mail legítimo. `real_lixeira` = % de e-mails reais marcados como spam. `spam_passou` = % de spam que chegou à caixa de entrada.
""")
C.extend(base)
code(r'''
# Os dois Stacking 7 (exp. 03/04) e o LinearSVC afinado, por indice no espaco de busca, conferidos por descricao.
PADRAO = tuple([0] * len(CARD))
melhor_ga = (1, 7, 1, 1, 5, 5, 4, 5)       # Especialista = GA do exp. 03 (#5 na corrida)
lsvc_best = 5                              # LinearSVC C=1 sem peso de classe
esperado = ["C=1, cw=balanced", "k=500, C=30", "k=3, w=uniform", "h=(64,), a=0.0001", "a=0.1, norm=True, usa_tf=False",
            "n=800, mf=sqrt, msl=1, cw=balanced", "n=400, eta=0.05, folhas=63"]
assert [desc(f, melhor_ga[i]) for i, f in enumerate(FAMILIAS)] == esperado and META_C[melhor_ga[-1]] == 10.0
assert desc("LinearSVC", lsvc_best) == "C=1, cw=None"
def meta(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
def t_corrigido(a, b, n_tr, n_te):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)
''')
C.extend(conf)

md("## 7. Notas do TabPFN e conferência das divisões")
code(r'''
S = np.load("splits_tabpfn.npz")
for c in CONF:   # o TabPFN usou exatamente os mesmos e-mails que o motor
    assert np.array_equal(dev[SPLITS[c]["tr"]], S[f"{c}__tr"]) and np.array_equal(dev[SPLITS[c]["tes"]["te"]], S[f"{c}__te"]), c
TAB = {}
for modo in ["nuvem", "local"]:
    for arq_ in sorted(Path(f"cache_tabpfn/{modo}").glob("*.npz")):
        if not arq_.stem.endswith("_tmp"):
            z = np.load(arq_); TAB[modo, arq_.stem] = {k: z[k] for k in z.files}
CONF_NUVEM = [c for c in CONF if ("nuvem", c) in TAB]
CONF_LOCAL = [c for c in CONF if ("local", c) in TAB]
log(f"notas do TabPFN | nuvem: {len(CONF_NUVEM)} divisões + externo={('nuvem', 'externo') in TAB} | local: {len(CONF_LOCAL)} divisões")
''')

md("## 8. Os seis métodos nas 20 divisões")
code(r'''
def prever_tab(modo, chave, te):
    """TabPFN sozinho: (nota, decisao 0,5, decisao no limiar de custo escolhido com a nota OOF do treino)."""
    sp = SPLITS[chave]; ytr = BASES[sp["base"]][1][sp["tr"]]; z = TAB[modo, chave]
    p = z[f"zte_{te}"]
    return p, (p > 0.5).astype(int), (p > limiar_custo(z["ztr"], ytr)).astype(int)

def prever_stack8(gen, modo, chave, te):
    """Stacking 8 = os 7 do cromossomo + TabPFN como 8a coluna do meta-modelo."""
    sp = SPLITS[chave]; ytr = BASES[sp["base"]][1][sp["tr"]]
    par = [carregar(chave, fam, gen[i], te) for i, fam in enumerate(FAMILIAS)]
    z = TAB[modo, chave]
    Ztr = np.column_stack([a for a, _ in par] + [z["ztr"]]); Zte = np.column_stack([b for _, b in par] + [z[f"zte_{te}"]])
    m = meta(META_C[gen[-1]])
    oof = cross_val_predict(m, Ztr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    m.fit(Ztr, ytr); p = m.predict_proba(Zte)[:, 1]
    return p, (p > 0.5).astype(int), (p > limiar_custo(oof, ytr)).astype(int), m[-1].coef_.ravel()

M6 = ["TabPFN-3.5-Plus", "Stacking 7 padrão", "Stacking 7 especialista", "Stacking 8 padrão (+TabPFN)",
      "Stacking 8 especialista (+TabPFN)", "LinearSVC otimizado"]
METODOS["Stacking 7 especialista"] = ("stack", melhor_ga)
def prever6(nome, chave, te, modo="nuvem"):
    if nome.startswith("TabPFN"): return prever_tab(modo, chave, te), None
    if nome.startswith("Stacking 8"):
        *r, w = prever_stack8(PADRAO if "padrão" in nome else melhor_ga, modo, chave, te); return tuple(r), w
    return prever("Stacking 7 otimizado (GA)" if nome == "Stacking 7 especialista" else nome, chave, te), None

linhas, pesos8 = [], []
for c in CONF_NUVEM:
    yte = y[SPLITS[c]["tes"]["te"]]
    for nome in M6:
        r, w = prever6(nome, c, "te")
        linhas.append(dict(metodo=nome, split=c, n_tr=len(SPLITS[c]["tr"]), n_te=len(yte), **metricas(yte, *r)))
        if w is not None: pesos8.append(dict(metodo=nome, split=c, **dict(zip(FAMILIAS + ["TabPFN"], w))))
R5 = pd.DataFrame(linhas); R5.to_csv("tabpfn_20folds.csv", sep=";", decimal=",", index=False)
res5 = R5.groupby("metodo")[["mcc", "pr_auc", "real_lixeira", "spam_passou", "mcc_custo", "real_lixeira_custo", "spam_passou_custo", "custo_por_mil"]].mean()
res5[["real_lixeira", "spam_passou", "real_lixeira_custo", "spam_passou_custo"]] *= 100
print(f"{len(CONF_NUVEM)} divisões (sementes 201–204)")
res5.loc[M6].round(4)
''')
code(r'''
rho = R5.n_te.mean() / R5.n_tr.mean()
def p_melhor(d):
    d = np.asarray(d); n = len(d); s = np.sqrt((1 / n + rho) * d.var(ddof=1))
    return float(1 - stats.t.cdf(0, n - 1, loc=d.mean(), scale=s)) if s > 0 else float(d.mean() > 0)
PARES5 = [("Stacking 8 padrão (+TabPFN)", "Stacking 7 padrão"), ("Stacking 8 especialista (+TabPFN)", "Stacking 7 especialista"),
          ("TabPFN-3.5-Plus", "LinearSVC otimizado"), ("Stacking 7 padrão", "TabPFN-3.5-Plus"),
          ("Stacking 8 especialista (+TabPFN)", "Stacking 7 padrão")]
lin = []
for met, sinal in [("mcc", 1), ("mcc_custo", 1), ("custo_por_mil", -1)]:
    P = R5.pivot(index="split", columns="metodo", values=met)
    for a_, b_ in PARES5:
        _, _, p = t_corrigido(P[a_], P[b_], R5.n_tr.mean(), R5.n_te.mean())
        eq = np.nan
        if met != "custo_por_mil":
            try: eq = baycomp.two_on_single(P[a_].values, P[b_].values, rope=0.01, runs=len(CONF_NUVEM) // 5)[1]
            except Exception: pass
        lin.append(dict(metrica=met, A=a_, B=b_, A_media=P[a_].mean(), B_media=P[b_].mean(), p=p,
                        **{"P(A melhor)": p_melhor(sinal * (P[a_] - P[b_])), "P(equiv.)": eq}))
print("P(A melhor) = probabilidade de A ser melhor por qualquer margem; P(equiv.) = |dif MCC| < 0,01")
pd.DataFrame(lin).round(4)
''')

md("## 9. O TabPFN erra em e-mails diferentes? Pesos no meta-modelo e erros em comum")
code(r'''
W8 = pd.DataFrame(pesos8)
print("peso médio de cada família no meta-modelo do Stacking 8 (unidades padronizadas):")
print(W8.groupby("metodo")[FAMILIAS + ["TabPFN"]].mean().T.round(2).to_string())

# erros em comum nas notas out-of-fold do treino (todas as 20 divisoes juntas)
err = {n: [] for n in FAMILIAS + ["TabPFN"]}; ys = []
for c in CONF_NUVEM:
    ytr = y[SPLITS[c]["tr"]]; ys.append(ytr)
    for i, fam in enumerate(FAMILIAS):
        z = carregar(c, fam, 0, "te")[0]; lim = 0.0 if fam == "SVM RBF" else 0.5   # SVC sem proba: nota = distancia a margem
        err[fam].append((z > lim) != ytr)
    err["TabPFN"].append((TAB["nuvem", c]["ztr"] > 0.5) != ytr)
E_ = pd.DataFrame({n: np.concatenate(v) for n, v in err.items()})
so_tab = (E_["TabPFN"] & ~E_[FAMILIAS].any(axis=1)).sum(); tab_salva = (~E_["TabPFN"] & E_[FAMILIAS].sum(axis=1).ge(4)).sum()
print(f"\nerros do TabPFN: {E_['TabPFN'].sum()} | só o TabPFN erra: {so_tab} | TabPFN acerta onde a maioria (≥4 de 7) erra: {tab_salva}")
print("taxa de erro em comum (Jaccard) do TabPFN com cada família (configuração padrão):")
print(pd.Series({f: (E_[f] & E_["TabPFN"]).sum() / (E_[f] | E_["TabPFN"]).sum() for f in FAMILIAS}).round(3).sort_values().to_string())
''')

md("## 10. Base externa: Enron2–6 (treino nos 5172 e-mails)")
code(r'''
if ("nuvem", "externo") in TAB:
    EXT = next((p for p in [Path("../../dados/enron_vetores.npz"), Path("../dados_externos/enron_vetores.npz")] if p.exists()), None)
    z = np.load(EXT); yext = {f"Enron{k}": z[f"y{k}"][S[f"ext_Enron{k}"]] for k in range(2, 7)}
    SPLITS["externo"] = dict(base="all", tr=np.arange(len(y_all)), tes={b: None for b in yext}, si=SEED)
    lin = []
    for b in yext:
        for nome in M6:
            r, _ = prever6(nome, "externo", b)
            lin.append(dict(base=b, metodo=nome, **metricas(yext[b], *r)))
    E5 = pd.DataFrame(lin); E5.to_csv("tabpfn_externo.csv", sep=";", decimal=",", index=False)
    for col, nome, pc in [("mcc_custo", "MCC, limiar de custo", False), ("pr_auc", "PR-AUC", False), ("custo_por_mil", "custo por mil e-mails, limiar de custo", False),
                          ("real_lixeira_custo", "% de e-mails reais na lixeira, limiar de custo", True), ("spam_passou_custo", "% de spam que passou, limiar de custo", True)]:
        t = E5.pivot(index="metodo", columns="base", values=col).reindex(M6); t["média"] = t.mean(1)
        print(f"\n== {nome} ==\n" + (t * 100 if pc else t).round(1 if (pc or col == "custo_por_mil") else 3).to_string())
else:
    E5 = None; print("notas externas do TabPFN ainda não geradas")
''')

md("## 11. Réplica local (TabPFN-3.5-Fast em CPU) × nuvem (3.5-Plus), nas mesmas divisões")
code(r'''
if CONF_LOCAL:
    lin = []
    for c in CONF_LOCAL:
        yte = y[SPLITS[c]["tes"]["te"]]
        for modo, rot in [("nuvem", "TabPFN-3.5-Plus (nuvem)"), ("local", "TabPFN-3.5-Fast (local)")]:
            lin.append(dict(metodo=rot, split=c, **metricas(yte, *prever_tab(modo, c, "te"))))
            *r, _ = prever_stack8(PADRAO, modo, c, "te")
            lin.append(dict(metodo=f"Stacking 8 padrão + {rot}", split=c, **metricas(yte, *r)))
        lin.append(dict(metodo="Stacking 7 padrão", split=c, **metricas(yte, *prever("Stacking 7 padrão", c, "te"))))
    RL = pd.DataFrame(lin); RL.to_csv("tabpfn_local_x_nuvem.csv", sep=";", decimal=",", index=False)
    print(f"{len(CONF_LOCAL)} divisões (semente 201)")
    print(RL.groupby("metodo")[["mcc", "pr_auc", "mcc_custo", "custo_por_mil"]].mean().round(4).to_string())
    corr = np.mean([np.corrcoef(TAB["local", c]["zte_te"], TAB["nuvem", c]["zte_te"])[0, 1] for c in CONF_LOCAL])
    print(f"\ncorrelação média entre as notas local e nuvem nos e-mails de teste: {corr:.3f}")
else:
    print("réplica local ainda não executada (rodar experimentos/rodar_noite_tabpfn.ps1)")
log("fim")
''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1]); print("ok", sys.argv[1], "|", len(C), "células")
