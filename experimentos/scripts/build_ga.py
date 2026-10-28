import nbformat as nbf, sys
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Experimento 03: otimização genética do Stacking 7

**Hipótese:** ajustar os hiperparâmetros dos 7 modelos de base e do meta-modelo melhora o Stacking 7 de forma real, e não só nas divisões usadas na busca.

**Termos.** *E-mail real* = e-mail legítimo. `real_lixeira` = % de e-mails reais marcados como spam. `spam_passou` = % de spam que chegou à caixa de entrada.

**Protocolo**
1. **Espaço de busca (a "árvore" de parâmetros prováveis):** de 6 a 8 configurações por família, partindo das recomendações da literatura e dos padrões das bibliotecas. A configuração 0 de cada família é a usada no experimento 02.
2. **Busca:** cada configuração é treinada uma única vez em 10 divisões (2 sementes × 5 folds, só com os 80% de desenvolvimento) e as notas *out-of-fold* ficam em cache. Assim, avaliar um indivíduo do algoritmo genético é só treinar o meta-modelo (milissegundos), e a busca sai barata.
3. **Algoritmo genético:** cada cromossomo tem 8 genes (a configuração de cada uma das 7 famílias + o C do meta-modelo). Usa torneio, cruzamento uniforme, mutação e elitismo. A **busca aleatória com o mesmo número de avaliações** serve de controle.
4. **Confirmação com sementes novas** (4 × 5 = 20 folds que a busca nunca viu): Stacking 7 padrão × Stacking 7 otimizado × LinearSVC padrão × LinearSVC otimizado (o LinearSVC também é afinado, para a comparação ser justa). Os resultados aparecem com o **limiar padrão** e com o **limiar de custo** (1 e-mail real perdido = 9 spams recebidos; o limiar é escolhido só com o treino).
5. **Base de teste trancada** (2ª avaliação registrada) e **base externa Enron2–6** (cerca de 28 mil e-mails: a amostra grande que consegue separar diferenças pequenas).
""")

code(r'''
import os, gc, time, hashlib, platform, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import sparse, stats
from joblib import Parallel, delayed, parallel_config
import psutil, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score
from lightgbm import LGBMClassifier
import baycomp, sklearn

SMOKE = bool(os.environ.get("GA_SMOKE"))          # execucao reduzida so para testar o codigo
SEED, N_THREADS = 42, 4
CUSTO_FP, CUSTO_FN = 9, 1                          # 1 e-mail real perdido = 9 spams recebidos
CACHE = Path("cache_ga_smoke" if SMOKE else "cache_ga").resolve(); CACHE.mkdir(exist_ok=True)
LOG = Path("progresso_ga_smoke.log" if SMOKE else "progresso_ga.log")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

def log(msg):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh: fh.write(time.strftime("%H:%M:%S ") + msg + "\n")
log(f"inicio | python {platform.python_version()} | sklearn {sklearn.__version__} | SMOKE={SMOKE}")
''')

md("## 1. Dados e a mesma base de teste trancada do experimento 02")
code(r'''
CSV = next(p for r in [Path("../.."), Path(".."), Path(".")] for p in [r / "dados" / "emails.csv", r / "emails.csv", r / "emails (1).csv"] if p.exists())
df = pd.read_csv(CSV); y_all = df.pop("Prediction").values; df.pop("Email No.")
g_all = pd.util.hash_pandas_object(df, index=False).values
A = np.ascontiguousarray(df.values, dtype=np.int64)
hash_csv = {hashlib.md5(r.tobytes()).digest() for r in A}
X_all = sparse.csr_matrix(A.astype(np.float32)); del df, A; gc.collect()

ix = np.load(next(p for p in [Path("../indices_teste.npz"), Path("indices_teste.npz")] if p.exists()))
dev, teste = ix["dev"], ix["teste"]
assert len(dev) + len(teste) == len(y_all) and not set(g_all[dev]) & set(g_all[teste])
X, y, g = X_all[dev], y_all[dev], g_all[dev]
BASES = {"dev": (X, y, g), "all": (X_all, y_all, g_all)}
log(f"desenvolvimento {len(dev)} | teste trancado {len(teste)} ({(y_all[teste] == 0).sum()} reais, {(y_all[teste] == 1).sum()} spam)")
''')

md("## 2. Espaço de busca: a árvore de parâmetros prováveis")
code(r'''
tf = lambda: TfidfTransformer(sublinear_tf=True)
def lr(C, cw):          return make_pipeline(tf(), LogisticRegression(C=C, class_weight=cw, max_iter=5000))
def svm(k, C):          return make_pipeline(tf(), TruncatedSVD(k, random_state=SEED), Normalizer(), SVC(C=C, class_weight="balanced"))
def knn(k, w):          return make_pipeline(tf(), KNeighborsClassifier(k, weights=w, metric="cosine"))
def mlp(h, a):          return make_pipeline(tf(), MLPClassifier(h, alpha=a, early_stopping=True, max_iter=200, random_state=SEED))
def cnb(a, norm, usa_tf):
    return make_pipeline(tf(), ComplementNB(alpha=a, norm=norm)) if usa_tf else ComplementNB(alpha=a, norm=norm)
def et(n, mf, msl, cw): return ExtraTreesClassifier(n, max_features=mf, min_samples_leaf=msl, class_weight=cw, n_jobs=N_THREADS, random_state=SEED)
def lgbm(n, eta, folhas, col=1.0, sub=1.0, mcs=20):
    return LGBMClassifier(n_estimators=n, learning_rate=eta, num_leaves=folhas, colsample_bytree=col, subsample=sub,
                          subsample_freq=1 if sub < 1 else 0, min_child_samples=mcs, n_jobs=N_THREADS, verbose=-1, random_state=SEED)
def lsvc(C, cw):        return make_pipeline(tf(), LinearSVC(C=C, class_weight=cw))

ESPACO = {   # posicao 0 = configuracao do experimento 02
  "Reg. Logística": [(lr, dict(C=C, cw=cw)) for C, cw in
                     [(10, "balanced"), (1, "balanced"), (3, "balanced"), (30, "balanced"), (100, "balanced"), (10, None), (1, None), (100, None)]],
  "SVM RBF":        [(svm, dict(k=k, C=C)) for k, C in [(300, 10), (300, 3), (300, 30), (150, 10), (150, 30), (500, 10), (300, 1), (500, 30)]],
  "kNN cosseno":    [(knn, dict(k=k, w=w)) for k, w in
                     [(5, "uniform"), (3, "uniform"), (7, "uniform"), (11, "uniform"), (15, "uniform"), (5, "distance"), (11, "distance"), (21, "distance")]],
  "MLP":            [(mlp, dict(h=h, a=a)) for h, a in [((128,), 1e-4), ((64,), 1e-4), ((256,), 1e-4), ((128,), 1e-3), ((128,), 1e-2), ((256, 128), 1e-4)]],
  "Naive Bayes":    [(cnb, dict(a=a, norm=n, usa_tf=t)) for a, n, t in
                     [(1, False, False), (0.1, False, False), (0.3, False, False), (3, False, False), (1, True, False), (0.1, True, False), (0.1, False, True), (1, False, True)]],
  "Extra Trees":    [(et, dict(n=n, mf=mf, msl=msl, cw=cw)) for n, mf, msl, cw in
                     [(400, "sqrt", 1, "balanced"), (400, "sqrt", 1, None), (400, "log2", 1, "balanced"), (400, 0.05, 1, "balanced"), (400, "sqrt", 2, "balanced"), (800, "sqrt", 1, "balanced")]],
  "LightGBM":       [(lgbm, kw) for kw in
                     [dict(n=400, eta=0.05, folhas=31), dict(n=200, eta=0.1, folhas=31), dict(n=800, eta=0.03, folhas=31), dict(n=400, eta=0.05, folhas=15),
                      dict(n=400, eta=0.05, folhas=63), dict(n=400, eta=0.05, folhas=31, col=0.3), dict(n=800, eta=0.05, folhas=15, col=0.3, sub=0.8),
                      dict(n=400, eta=0.05, folhas=31, mcs=5)]],
}
LINEARSVC = [(lsvc, dict(C=C, cw=cw)) for C, cw in [(1, "balanced"), (0.1, "balanced"), (0.3, "balanced"), (3, "balanced"), (10, "balanced"), (1, None), (0.3, None), (3, None)]]
META_C = [1.0, 0.03, 0.1, 0.3, 3.0, 10.0]
if SMOKE:
    ESPACO = {f: v[:2] for f, v in ESPACO.items()}; LINEARSVC = LINEARSVC[:2]; META_C = META_C[:2]
FAMILIAS = list(ESPACO)
TODOS = {**ESPACO, "LinearSVC": LINEARSVC}
FID = {f: i for i, f in enumerate(TODOS)}
CARD = [len(ESPACO[f]) for f in FAMILIAS] + [len(META_C)]
desc = lambda fam, k: ", ".join(f"{a}={b}" for a, b in TODOS[fam][k][1].items())

print(pd.DataFrame([dict(familia=f, opcoes=len(v), padrao=desc(f, 0)) for f, v in TODOS.items()]
                   + [dict(familia="meta-modelo (C)", opcoes=len(META_C), padrao=f"C={META_C[0]}")]).to_string(index=False))
print(f"\ncombinações possíveis do Stacking 7: {int(np.prod(CARD)):,}".replace(",", "."))
''')

md("## 3. Motor: treina cada configuração uma vez por divisão e guarda as notas em cache")
code(r'''
def nota(m, A):
    return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)

def arq(chave, fam, k):
    return CACHE / f"{chave}__f{FID[fam]}__k{k}.npz"

def tarefa(chave, Xtr, ytr, gtr, tes, si, fam, k):
    """Nota out-of-fold no treino (CV interna por grupo, igual para todas as configuracoes) + nota em cada teste."""
    destino = arq(chave, fam, k)
    if destino.exists(): return
    f, kw = TODOS[fam][k]; t0 = time.time(); ztr = np.zeros(Xtr.shape[0])
    for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=si).split(Xtr, ytr, gtr):
        m = f(**kw).fit(Xtr[itr], ytr[itr]); ztr[ite] = nota(m, Xtr[ite])
    m = f(**kw).fit(Xtr, ytr)
    tmp = destino.with_name(destino.stem + "_tmp.npz")
    np.savez(tmp, ztr=ztr, t=time.time() - t0, **{f"zte_{n}": nota(m, A) for n, A in tes.items()})
    os.replace(tmp, destino)

SPLITS = {}   # chave -> dict(base, tr, tes, si)
def args(chave, fam, k):
    sp = SPLITS[chave]; Xb, yb, gb = BASES[sp["base"]]; tr = sp["tr"]
    tes = {n: (Xb[v] if isinstance(v, np.ndarray) and v.ndim == 1 else v) for n, v in sp["tes"].items()}
    return chave, Xb[tr], yb[tr], gb[tr], tes, sp["si"], fam, k

LENTO = {"Extra Trees": 0, "LightGBM": 1, "MLP": 2, "SVM RBF": 3}
def rodar(tarefas, rotulo, blocos=8):
    """Executa em blocos; a cada bloco escolhe quantos processos usar pela RAM livre."""
    global N_THREADS
    pend = sorted([t for t in tarefas if not arq(*t).exists()], key=lambda t: LENTO.get(t[1], 9))
    log(f"{rotulo}: {len(tarefas)} tarefas, {len(pend)} pendentes")
    t_ini = time.time()
    for q in range(blocos):
        bloco = pend[q::blocos]
        if not bloco: continue
        livre = psutil.virtual_memory().available / 2**30
        nw = 4 if livre > 5 else 3 if livre > 3.5 else 2 if livre > 2.5 else 1
        N_THREADS = 4 if nw == 1 else 2; t0 = time.time()
        if nw == 1:
            for t in bloco: tarefa(*args(*t))
        else:
            with parallel_config(backend="loky", inner_max_num_threads=N_THREADS):
                Parallel(n_jobs=nw)(delayed(tarefa)(*args(*t)) for t in bloco)
        gc.collect()
        log(f"{rotulo}: bloco {q + 1}/{blocos} ok | {len(bloco)} tarefas | {nw} em paralelo | RAM livre {livre:.1f} GB | "
            f"{time.time() - t0:.0f}s (total {time.time() - t_ini:.0f}s)")

def carregar(chave, fam, k, te):
    z = np.load(arq(chave, fam, k)); return z["ztr"], z[f"zte_{te}"]

def mcc_np(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    tp = np.sum(a & b); tn = np.sum(~a & ~b); fp = np.sum(~a & b); fn = np.sum(a & ~b)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0
''')

md("## 4. Busca: 10 divisões (2 sementes × 5 folds) com todas as configurações")
code(r'''
BUSCA_SEEDS = [11] if SMOKE else [11, 12]
for s in BUSCA_SEEDS:
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g)):
        if SMOKE and f >= 2: break
        SPLITS[f"b{s}_{f}"] = dict(base="dev", tr=tr, tes={"te": te}, si=s * 100 + f)
BUSCA = [c for c in SPLITS if c.startswith("b")]
rodar([(c, fam, k) for c in BUSCA for fam in TODOS for k in range(len(TODOS[fam]))], "busca")

ZB = {c: {fam: [carregar(c, fam, k, "te") for k in range(len(TODOS[fam]))] for fam in TODOS} for c in BUSCA}
YB = {c: (y[SPLITS[c]["tr"]], y[SPLITS[c]["tes"]["te"]]) for c in BUSCA}
tempo = pd.Series({f"{fam} | {desc(fam, k)}": np.mean([float(np.load(arq(c, fam, k))["t"]) for c in BUSCA])
                   for fam in TODOS for k in range(len(TODOS[fam]))})
print("tempo médio por configuração e divisão (6 treinos), as 8 mais lentas:")
print(tempo.sort_values(ascending=False).head(8).round(1).to_string())
''')

md("## 5. Algoritmo genético × busca aleatória (mesmo orçamento)")
code(r'''
AVAL = {}   # memoria: cromossomo -> MCC medio nas 10 divisoes de busca
def meta(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
def aptidao(gen):
    gen = tuple(int(v) for v in gen)
    if gen not in AVAL:
        v = []
        for c in BUSCA:
            Ztr = np.column_stack([ZB[c][fam][gen[i]][0] for i, fam in enumerate(FAMILIAS)])
            Zte = np.column_stack([ZB[c][fam][gen[i]][1] for i, fam in enumerate(FAMILIAS)])
            p = meta(META_C[gen[-1]]).fit(Ztr, YB[c][0]).predict_proba(Zte)[:, 1]
            v.append(mcc_np(YB[c][1], p > 0.5))
        AVAL[gen] = float(np.mean(v))
    return AVAL[gen]

PADRAO = tuple([0] * len(CARD))
def algoritmo_genetico(n_pop, n_ger, p_cruz=0.9, p_mut=0.15, elite=2, seed=SEED):
    rng = np.random.default_rng(seed)
    pop = [PADRAO] + [tuple(int(rng.integers(c)) for c in CARD) for _ in range(n_pop - 1)]
    hist = []
    for ger in range(n_ger):
        apt = np.array([aptidao(i) for i in pop]); ordem = np.argsort(-apt)
        hist.append(dict(geracao=ger, melhor=apt[ordem[0]], media=apt.mean(), avaliacoes=len(AVAL)))
        def torneio():
            c = rng.choice(len(pop), 3, replace=False); return pop[c[np.argmax(apt[c])]]
        nova = [pop[i] for i in ordem[:elite]]
        while len(nova) < n_pop:
            a, b = torneio(), torneio()
            filho = tuple(x if m else z for x, z, m in zip(a, b, rng.random(len(CARD)) < 0.5)) if rng.random() < p_cruz else a
            nova.append(tuple(int(rng.integers(c)) if rng.random() < p_mut else x for x, c in zip(filho, CARD)))
        pop = nova
    apt = np.array([aptidao(i) for i in pop])
    return pop[int(np.argmax(apt))], pd.DataFrame(hist)

t0 = time.time()
melhor_ga, hist_ga = algoritmo_genetico(*((6, 3) if SMOKE else (40, 30)))
seq_ga = list(AVAL.values()); n_orc = len(seq_ga)
rng = np.random.default_rng(SEED + 1)
seq_al = [aptidao(tuple(int(rng.integers(c)) for c in CARD)) for _ in range(n_orc)]
log(f"GA: {n_orc} avaliações únicas em {time.time() - t0:.0f}s")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(np.maximum.accumulate(seq_ga), label="algoritmo genético")
ax.plot(np.maximum.accumulate(seq_al), label="busca aleatória")
ax.axhline(aptidao(PADRAO), color="gray", ls="--", label="Stacking 7 padrão")
ax.set_xlabel("avaliações"); ax.set_ylabel("melhor MCC (busca)"); ax.legend(); ax.set_title("Melhor encontrado × número de avaliações")
plt.tight_layout(); plt.show()

print(f"MCC na busca | padrão {aptidao(PADRAO):.4f} | melhor do GA {aptidao(melhor_ga):.4f} | melhor da busca aleatória {max(seq_al):.4f}")
print("\nconfiguração escolhida pelo GA:")
for i, fam in enumerate(FAMILIAS):
    print(f"  {fam:15s} {desc(fam, melhor_ga[i])}" + ("   (= padrão)" if melhor_ga[i] == 0 else ""))
print(f"  {'meta-modelo':15s} C={META_C[melhor_ga[-1]]}" + ("   (= padrão)" if melhor_ga[-1] == 0 else ""))
''')
code(r'''
# Sensibilidade: partindo do melhor do GA, troca um gene por vez. Mostra quais familias importam.
sens = []
for i, nome in enumerate(FAMILIAS + ["meta-modelo (C)"]):
    v = [aptidao(melhor_ga[:i] + (k,) + melhor_ga[i + 1:]) for k in range(CARD[i])]
    sens.append(dict(gene=nome, pior=min(v), melhor=max(v), amplitude=max(v) - min(v)))
print("quanto o MCC da busca varia trocando só a configuração de uma família:")
print(pd.DataFrame(sens).sort_values("amplitude", ascending=False).round(4).to_string(index=False))

# LinearSVC tambem e afinado (comparacao justa): melhor C nas mesmas 10 divisoes
mcc_lsvc = [np.mean([mcc_np(YB[c][1], ZB[c]["LinearSVC"][k][1] > 0) for c in BUSCA]) for k in range(len(LINEARSVC))]
lsvc_best = int(np.argmax(mcc_lsvc))
print(f"\nLinearSVC na busca | padrão ({desc('LinearSVC', 0)}) {mcc_lsvc[0]:.4f} | melhor ({desc('LinearSVC', lsvc_best)}) {mcc_lsvc[lsvc_best]:.4f}")
''')

md(r"""
## 6. Confirmação com sementes novas (20 folds que a busca não viu)
O ganho medido na própria busca é otimista: o GA escolheu o melhor de mais de mil candidatos nas mesmas 10 divisões. A medida honesta vem de divisões novas.
""")
code(r'''
METODOS = {"Stacking 7 padrão": ("stack", PADRAO), "Stacking 7 otimizado (GA)": ("stack", melhor_ga),
           "LinearSVC padrão": ("lsvc", 0), "LinearSVC otimizado": ("lsvc", lsvc_best)}
PRECISA = sorted({(fam, 0) for fam in FAMILIAS} | {(fam, melhor_ga[i]) for i, fam in enumerate(FAMILIAS)}
                 | {("LinearSVC", 0), ("LinearSVC", lsvc_best)}, key=str)

def limiar_custo(s, yv):
    """Limiar que minimiza 9 x (reais na lixeira) + 1 x (spams que passaram), escolhido so com o treino."""
    o = np.argsort(-s); ss, yy = s[o], yv[o]
    custo = CUSTO_FP * np.cumsum(yy == 0) + CUSTO_FN * ((yy == 1).sum() - np.cumsum(yy == 1))
    i = int(np.argmin(custo))
    if CUSTO_FN * (yy == 1).sum() <= custo[i]: return np.inf
    return (ss[i] + ss[i + 1]) / 2 if i + 1 < len(ss) else ss[i] - 1e-9

def prever(metodo, chave, te):
    """(nota, decisao no limiar padrao, decisao no limiar de custo)."""
    tipo, cfg = METODOS[metodo]; sp = SPLITS[chave]; ytr = BASES[sp["base"]][1][sp["tr"]]
    if tipo == "lsvc":
        ztr, zte = carregar(chave, "LinearSVC", cfg, te)
        return zte, (zte > 0).astype(int), (zte > limiar_custo(ztr, ytr)).astype(int)
    par = [carregar(chave, fam, cfg[i], te) for i, fam in enumerate(FAMILIAS)]
    Ztr, Zte = np.column_stack([a for a, _ in par]), np.column_stack([b for _, b in par])
    m = meta(META_C[cfg[-1]])
    oof = cross_val_predict(m, Ztr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    p = m.fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    return p, (p > 0.5).astype(int), (p > limiar_custo(oof, ytr)).astype(int)

def metricas(yv, s, yh, yc):
    fp, fn = int(((yc == 1) & (yv == 0)).sum()), int(((yc == 0) & (yv == 1)).sum())
    return dict(mcc=matthews_corrcoef(yv, yh), pr_auc=average_precision_score(yv, s),
                real_lixeira=(yh[yv == 0] == 1).mean(), spam_passou=(yh[yv == 1] == 0).mean(),
                mcc_custo=matthews_corrcoef(yv, yc), real_lixeira_custo=(yc[yv == 0] == 1).mean(),
                spam_passou_custo=(yc[yv == 1] == 0).mean(), custo_por_mil=1000 * (CUSTO_FP * fp + CUSTO_FN * fn) / len(yv))

CONF_SEEDS = [201] if SMOKE else [201, 202, 203, 204]
for s in CONF_SEEDS:
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g)):
        if SMOKE and f >= 2: break
        SPLITS[f"c{s}_{f}"] = dict(base="dev", tr=tr, tes={"te": te}, si=s * 100 + f)
CONF = [c for c in SPLITS if c.startswith("c")]
rodar([(c, fam, k) for c in CONF for fam, k in PRECISA], "confirmação", blocos=4)

linhas = []
for c in CONF:
    yte = y[SPLITS[c]["tes"]["te"]]
    for mt in METODOS:
        linhas.append(dict(metodo=mt, split=c, n_tr=len(SPLITS[c]["tr"]), n_te=len(yte), **metricas(yte, *prever(mt, c, "te"))))
R = pd.DataFrame(linhas); R.to_csv("ga_confirmacao.csv", sep=";", decimal=",", index=False)
pct = ["real_lixeira", "spam_passou", "real_lixeira_custo", "spam_passou_custo"]
res = R.groupby("metodo")[["mcc", "pr_auc", "real_lixeira", "spam_passou", "mcc_custo", "real_lixeira_custo", "spam_passou_custo", "custo_por_mil"]].mean()
res[pct] *= 100
res.loc[list(METODOS)].round(4)
''')
code(r'''
def t_corrigido(a, b, n_tr, n_te):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)

PARES = [("Stacking 7 otimizado (GA)", "Stacking 7 padrão"), ("Stacking 7 otimizado (GA)", "LinearSVC otimizado"),
         ("Stacking 7 padrão", "LinearSVC otimizado"), ("LinearSVC otimizado", "LinearSVC padrão")]
n_tr, n_te = R.n_tr.mean(), R.n_te.mean()
for met in ["mcc", "mcc_custo"]:
    P = R.pivot(index="split", columns="metodo", values=met)
    cmp_ = []
    for a, b in PARES:
        d, t, p = t_corrigido(P[a], P[b], n_tr, n_te)
        try:
            pa, pe, pb = baycomp.two_on_single(P[a].values, P[b].values, rope=0.01, runs=len(CONF_SEEDS))
        except Exception:
            pa = pe = pb = np.nan
        cmp_.append(dict(A=a, B=b, dif=d, p=p, P_A_melhor=pa, P_equivalentes=pe, P_B_melhor=pb))
    cmp_ = pd.DataFrame(cmp_).sort_values("p").reset_index(drop=True)
    cmp_["p_holm"] = np.minimum(1, np.maximum.accumulate((len(cmp_) - cmp_.index) * cmp_.p))
    print(f"\n== {met} (limiar {'padrão' if met == 'mcc' else 'de custo'}), {len(P)} folds; equivalentes = |dif| < 0,01 ==")
    print(cmp_[["A", "B", "dif", "p", "p_holm", "P_A_melhor", "P_equivalentes", "P_B_melhor"]].round(4).to_string(index=False))

ganho_busca = aptidao(melhor_ga) - aptidao(PADRAO)
ganho_conf = (R[R.metodo == "Stacking 7 otimizado (GA)"].mcc.values - R[R.metodo == "Stacking 7 padrão"].mcc.values).mean()
print(f"\nganho do GA sobre o padrão | na busca {ganho_busca:+.4f} | em sementes novas {ganho_conf:+.4f}  (a diferença é o otimismo da seleção)")
''')
code(r'''
# Mais sementes ajudam? O erro padrao corrigido tem um piso: as divisoes reusam os mesmos 4137 e-mails.
P = R.pivot(index="split", columns="metodo", values="mcc")
dp = (P["Stacking 7 otimizado (GA)"] - P["LinearSVC otimizado"]).std(ddof=1)
print(f"desvio da diferença entre folds: {dp:.4f} | fração de teste/treino: {n_te / n_tr:.2f}")
print(pd.DataFrame([dict(folds=J, erro_padrao=dp * np.sqrt(1 / J + n_te / n_tr), menor_dif_detectavel=2 * dp * np.sqrt(1 / J + n_te / n_tr))
                    for J in [5, 10, 15, 20, 30, 50, 100, 1000]]).round(4).to_string(index=False))
''')

md(r"""
## 7. Base de teste trancada (2ª avaliação registrada; nenhuma escolha é feita aqui)
Os quatro métodos são treinados nos 80% de desenvolvimento e avaliados uma vez nos 1035 e-mails trancados.
""")
code(r'''
SPLITS["teste"] = dict(base="all", tr=dev, tes={"teste": teste}, si=SEED)
rodar([("teste", fam, k) for fam, k in PRECISA], "teste trancado", blocos=2)
yt = y_all[teste]; PT = {mt: prever(mt, "teste", "teste") for mt in METODOS}
tab = pd.DataFrame([dict(metodo=mt, **metricas(yt, *PT[mt]),
                         reais_lixeira=int(((PT[mt][1] == 1) & (yt == 0)).sum()), spams_passaram=int(((PT[mt][1] == 0) & (yt == 1)).sum()),
                         reais_lixeira_custo=int(((PT[mt][2] == 1) & (yt == 0)).sum()), spams_passaram_custo=int(((PT[mt][2] == 0) & (yt == 1)).sum()))
                    for mt in METODOS]).set_index("metodo")
print(tab[["mcc", "pr_auc", "reais_lixeira", "spams_passaram", "mcc_custo", "reais_lixeira_custo", "spams_passaram_custo"]].round(4).to_string())

def mcnemar(acerto_a, acerto_b):
    na, nb = int((acerto_a & ~acerto_b).sum()), int((~acerto_a & acerto_b).sum())
    return na, nb, (stats.binomtest(min(na, nb), na + nb, 0.5).pvalue if na + nb else 1.0)
print()
for a, b in PARES[:3]:
    for j, rot in [(1, "padrão"), (2, "custo")]:
        na, nb, p = mcnemar(PT[a][j] == yt, PT[b][j] == yt)
        print(f"McNemar limiar {rot:6s} | {a} × {b}: só A acerta {na} | só B acerta {nb} | p = {p:.3f}")
''')

md("## 8. Base externa: Enron2–6 (cerca de 28 mil e-mails de outras caixas, nunca vistos; cópias do CSV removidas)")
code(r'''
EXT = next((p for p in [Path("../../dados/enron_vetores.npz"), Path("../dados_externos/enron_vetores.npz"), Path("enron_vetores.npz")] if p.exists()), None)
E = None
if EXT is not None:
    z = np.load(EXT); Xext, yext = {}, {}
    for k in range(2, 7):
        A = z[f"X{k}"].astype(np.int64)
        novo = np.array([hashlib.md5(r.tobytes()).digest() not in hash_csv for r in A])
        Xext[f"Enron{k}"] = sparse.csr_matrix(A[novo].astype(np.float32)); yext[f"Enron{k}"] = z[f"y{k}"][novo]
    del A; gc.collect()
    SPLITS["externo"] = dict(base="all", tr=np.arange(len(y_all)), tes=Xext, si=SEED)
    rodar([("externo", fam, k) for fam, k in PRECISA], "externo", blocos=2)
    lin, PE = [], {}
    for b in Xext:
        for mt in METODOS:
            PE[b, mt] = prever(mt, "externo", b)
            lin.append(dict(base=b, metodo=mt, **metricas(yext[b], *PE[b, mt])))
    E = pd.DataFrame(lin); E.to_csv("ga_externo.csv", sep=";", decimal=",", index=False)
    for col, nome, pc in [("mcc", "MCC, limiar padrão", False), ("mcc_custo", "MCC, limiar de custo", False), ("pr_auc", "PR-AUC (não depende do limiar)", False),
                          ("real_lixeira_custo", "% de e-mails reais na lixeira, limiar de custo", True), ("spam_passou_custo", "% de spam que passou, limiar de custo", True)]:
        t = E.pivot(index="metodo", columns="base", values=col).reindex(list(METODOS)); t["média"] = t.mean(1)
        print(f"\n== {nome} ==\n" + (t * 100 if pc else t).round(1 if pc else 3).to_string())
    print()
    yy = np.concatenate([yext[b] for b in Xext])
    for a, b2 in PARES[:3]:
        for j, rot in [(1, "padrão"), (2, "custo")]:
            ca = np.concatenate([PE[b, a][j] for b in Xext]) == yy; cb = np.concatenate([PE[b, b2][j] for b in Xext]) == yy
            na, nb, p = mcnemar(ca, cb)
            print(f"McNemar externo (todas as bases) limiar {rot:6s} | {a} × {b2}: só A acerta {na} | só B acerta {nb} | p = {p:.2g}")
else:
    print("enron_vetores.npz não encontrado: validação externa pulada")
''')

md("## 9. Visão única")
code(r'''
visao = pd.DataFrame({
    "MCC confirmação (20 folds)": res["mcc"], "MCC conf. limiar custo": res["mcc_custo"],
    "MCC teste trancado": tab["mcc"], "MCC teste trancado, custo": tab["mcc_custo"],
    "MCC externo (média)": E.groupby("metodo").mcc.mean() if E is not None else np.nan,
    "MCC externo, custo (média)": E.groupby("metodo").mcc_custo.mean() if E is not None else np.nan,
    "% reais na lixeira externo, custo": E.groupby("metodo").real_lixeira_custo.mean() * 100 if E is not None else np.nan,
    "% spam passou externo, custo": E.groupby("metodo").spam_passou_custo.mean() * 100 if E is not None else np.nan,
}).reindex(list(METODOS))
log("fim")
visao.round(3)
''')

if __name__ == "__main__":   # build_corrida.py importa as celulas daqui sem gravar este notebook
    nb["cells"] = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, sys.argv[1]); print("ok", sys.argv[1])
