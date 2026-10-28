import nbformat as nbf, sys
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Stacking ampliado + base de teste aleatória travada (rascunho técnico)

**Termos.** *E-mail real* = e-mail legítimo, que não é spam. `real_lixeira` = % de e-mails reais marcados como spam. `spam_passou` = % de spam que chegou à caixa de entrada.

**Protocolo**
1. Sorteia e **trava 20% dos e-mails como base de teste** (estratificada; cópias idênticas sempre do mesmo lado). Ela não é tocada até o passo 4.
2. Nos 80% restantes: validação cruzada 3×5 por grupo. Dentro de cada treino, uma segunda validação cruzada (também por grupo) gera as notas *out-of-fold* que alimentam os meta-modelos.
3. Compara 13 modelos únicos e 6 formas de combiná-los, com teste t corrigido + Holm e teste bayesiano.
4. Finalistas definidos por regra fixa, avaliados **uma única vez** na base de teste: IC95% por bootstrap e teste de McNemar.
5. Validação externa: treina nos 5172 e-mails e testa no Enron2–6 (outras caixas de e-mail, nunca vistas).
""")

code(r'''
import gc, time, hashlib, platform, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from scipy import sparse, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import baycomp, sklearn
SEED, SEED_TESTE, N_JOBS = 42, 123, 4
CACHE = Path("cache_stacking"); CACHE.mkdir(exist_ok=True)
LOG = Path("progresso_stacking.log")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(platform.python_version(), "sklearn", sklearn.__version__)
''')

md("## 1. Dados e base de teste travada")
code(r'''
CSV = next(p for p in [Path("../../dados/emails.csv"), Path("emails.csv"), Path("emails (1).csv"), Path("../../emails (1).csv")] if p.exists())
df = pd.read_csv(CSV); y_all = df.pop("Prediction").values; df.pop("Email No.")
g_all = pd.util.hash_pandas_object(df, index=False).values          # mesmo texto = mesmo grupo
A = np.ascontiguousarray(df.values, dtype=np.int64)
hash_csv = {hashlib.md5(r.tobytes()).digest() for r in A}             # usado para limpar a base externa
X_all = sparse.csr_matrix(A.astype(np.float32)); del df, A; gc.collect()

dev, teste = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED_TESTE).split(X_all, y_all, g_all))
assert not set(g_all[dev]) & set(g_all[teste]), "texto repetido dos dois lados"
np.savez("indices_teste.npz", dev=dev, teste=teste)
X, y, g = X_all[dev], y_all[dev], g_all[dev]
for nome, ix in [("desenvolvimento (80%)", dev), ("teste travado (20%)", teste)]:
    print(f"{nome:22s} {len(ix):5d} e-mails | spam {y_all[ix].mean():.1%} | {len(set(g_all[ix]))} textos distintos")
''')

md("## 2. Os 13 modelos de base\nCatBoost e HistGradientBoosting ficam de fora: repetem o LightGBM/XGBoost e custam 5–20× mais.")
code(r'''
class NBFeatures(BaseEstimator, TransformerMixin):
    """NBSVM (Wang & Manning 2012): binariza e pondera cada palavra pela log-razao do Naive Bayes."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        B = (X > 0).astype(np.float32)
        p = self.alpha + np.asarray(B[y == 1].sum(0)).ravel(); q = self.alpha + np.asarray(B[y == 0].sum(0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        return self
    def transform(self, X):
        B = (X > 0).astype(np.float32)
        return B.multiply(self.r_).tocsr() if sparse.issparse(B) else B * self.r_

tfidf = lambda: TfidfTransformer(sublinear_tf=True)
BASE = {
  "Reg. Logística":     lambda: make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
  "LinearSVC":          lambda: make_pipeline(tfidf(), LinearSVC(C=1, class_weight="balanced")),
  "SGD (huber)":        lambda: make_pipeline(tfidf(), SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=SEED)),
  "Ridge":              lambda: make_pipeline(tfidf(), RidgeClassifier(class_weight="balanced")),
  "NBSVM":              lambda: make_pipeline(NBFeatures(), LogisticRegression(C=1, max_iter=3000)),
  "SVM RBF (SVD300)":   lambda: make_pipeline(tfidf(), TruncatedSVD(300, random_state=SEED), Normalizer(), SVC(C=10, class_weight="balanced")),
  "kNN cosseno":        lambda: make_pipeline(tfidf(), KNeighborsClassifier(5, metric="cosine")),
  "MLP (rede neural)":  lambda: make_pipeline(tfidf(), MLPClassifier((128,), early_stopping=True, max_iter=200, random_state=SEED)),
  "Naive Bayes Compl.": lambda: ComplementNB(),
  "Random Forest":      lambda: RandomForestClassifier(400, n_jobs=N_JOBS, class_weight="balanced", random_state=SEED),
  "Extra Trees":        lambda: ExtraTreesClassifier(400, n_jobs=N_JOBS, class_weight="balanced", random_state=SEED),
  "LightGBM":           lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, n_jobs=N_JOBS, verbose=-1, random_state=SEED),
  "XGBoost":            lambda: XGBClassifier(n_estimators=400, learning_rate=0.05, tree_method="hist", n_jobs=N_JOBS, random_state=SEED),
}
COLS = list(BASE)
def nota(m, A):
    return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)
LIMIAR0 = {n: 0.5 if hasattr(BASE[n](), "predict_proba") else 0.0 for n in BASE}   # equivale ao predict() de cada um

def gerar_notas(Xtr, ytr, gtr, Xtes, chave, seed_interno):
    """Nota out-of-fold de cada modelo no treino (CV interna por grupo) + nota em cada base de teste
    (modelo refeito no treino inteiro). Fica em cache: rodar de novo nao retreina."""
    arq = CACHE / f"{chave}.npz"
    if arq.exists():
        z = np.load(arq)
        return z["Ztr"], {k[4:]: z[k] for k in z.files if k.startswith("Zte_")}, z["t_int"], z["t_fin"]
    interno = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed_interno).split(Xtr, ytr, gtr))
    Ztr = np.zeros((Xtr.shape[0], len(COLS))); Zte = {k: np.zeros((A.shape[0], len(COLS))) for k, A in Xtes.items()}
    t_int, t_fin = np.zeros(len(COLS)), np.zeros(len(COLS))
    for j, n in enumerate(COLS):
        t0 = time.time()
        for itr, ite in interno:
            m = BASE[n]().fit(Xtr[itr], ytr[itr]); Ztr[ite, j] = nota(m, Xtr[ite])
        t1 = time.time(); m = BASE[n]().fit(Xtr, ytr); t_int[j], t_fin[j] = t1 - t0, time.time() - t1
        for k, A in Xtes.items():
            Zte[k][:, j] = nota(m, A)
        del m; gc.collect()
    np.savez(arq, Ztr=Ztr, t_int=t_int, t_fin=t_fin, **{f"Zte_{k}": v for k, v in Zte.items()})
    return Ztr, Zte, t_int, t_fin
''')

md(r"""
## 3. Seis formas de combinar
| Combinação | Como decide |
|---|---|
| Stacking 4 | a de antes (LR + Naive Bayes + RF + LightGBM), agora com CV interna por grupo |
| Stacking 7 | um representante por família: linear, kernel, vizinhos, rede neural, probabilístico, bagging, boosting |
| Stacking 13 | todos os modelos; a meta LR aprende os pesos |
| Stacking 13 + L1 | idem, mas a penalidade L1 zera os modelos inúteis (escolha automática, só com dados de treino) |
| Caruana 13 | seleção gulosa (Caruana et al. 2004, usada no AutoGluon e no auto-sklearn): adiciona, um a um, o modelo que mais reduz o erro da média |
| Média 13 | votação: média simples das 13 probabilidades (referência sem aprendizado de pesos) |
""")
code(r'''
def platt(Ztr, ytr, Zte):
    """Poe cada nota na escala de probabilidade, calibrando so com o treino (necessario para media e Caruana)."""
    Ptr, Pte = np.zeros_like(Ztr), np.zeros_like(Zte)
    for j in range(Ztr.shape[1]):
        c = LogisticRegression(C=1e4, max_iter=1000).fit(Ztr[:, [j]], ytr)
        Ptr[:, j], Pte[:, j] = c.predict_proba(Ztr[:, [j]])[:, 1], c.predict_proba(Zte[:, [j]])[:, 1]
    return Ptr, Pte

def meta_lr(l1=False):
    def f(Ztr, ytr, Zte, Ptr, Pte):
        est = (LogisticRegressionCV(Cs=10, l1_ratios=(1.0,), solver="saga", scoring="neg_log_loss",
                                    max_iter=5000, use_legacy_attributes=False) if l1 else LogisticRegression(max_iter=2000))
        m = make_pipeline(StandardScaler(), est).fit(Ztr, ytr)
        return m.predict_proba(Zte)[:, 1], m[-1].coef_.ravel()
    return f

def media(Ztr, ytr, Zte, Ptr, Pte):
    return Pte.mean(1), np.full(Pte.shape[1], 1 / Pte.shape[1])

def caruana(Ztr, ytr, Zte, Ptr, Pte, n_iter=50):
    soma, escolha = np.zeros(len(ytr)), []
    for k in range(1, n_iter + 1):
        perdas = [log_loss(ytr, np.clip((soma + Ptr[:, j]) / k, 1e-6, 1 - 1e-6)) for j in range(Ptr.shape[1])]
        j = int(np.argmin(perdas)); escolha.append(j); soma += Ptr[:, j]
    w = np.bincount(escolha, minlength=Ptr.shape[1]) / n_iter
    return Pte @ w, w

ix = lambda nomes: [COLS.index(n) for n in nomes]
S4 = ["Reg. Logística", "Naive Bayes Compl.", "Random Forest", "LightGBM"]
FAMILIA = ["Reg. Logística", "SVM RBF (SVD300)", "kNN cosseno", "MLP (rede neural)", "Naive Bayes Compl.", "Extra Trees", "LightGBM"]
COMBINADORES = {   # nome: (colunas usadas, funcao)
  "Stacking 4 (LR+NB+RF+LGBM)":  (ix(S4), meta_lr()),
  "Stacking 7 (1 por família)":  (ix(FAMILIA), meta_lr()),
  "Stacking 13 (todos)":         (ix(COLS), meta_lr()),
  "Stacking 13 + L1":            (ix(COLS), meta_lr(l1=True)),
  "Caruana 13":                  (ix(COLS), caruana),
  "Média 13 (votação)":          (ix(COLS), media),
}

def prever(nome, Ztr, ytr, Zte, Ptr, Pte):
    """Devolve (nota, decisao 0/1) de um modelo unico ou de uma combinacao."""
    if nome in COMBINADORES:
        c, f = COMBINADORES[nome]; p, w = f(Ztr[:, c], ytr, Zte[:, c], Ptr[:, c], Pte[:, c])
        return p, (p > 0.5).astype(int), dict(zip([COLS[i] for i in c], w))
    j = COLS.index(nome)
    return Zte[:, j], (Zte[:, j] > LIMIAR0[nome]).astype(int), {}

def metricas(yv, yh, s):
    return dict(mcc=matthews_corrcoef(yv, yh), pr_auc=average_precision_score(yv, s), f1=f1_score(yv, yh),
                real_lixeira=(yh[yv == 0] == 1).mean(), spam_passou=(yh[yv == 1] == 0).mean())
''')

md("## 4. Validação cruzada 3×5 por grupo nos 80% (com CV interna para os meta-modelos)")
code(r'''
REPETICOES = [42, 7, 2026]
METODOS = COLS + list(COMBINADORES)
linhas, pesos = [], []
for s in REPETICOES:
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g)):
        t0 = time.time()
        Ztr, Zte, t_int, t_fin = gerar_notas(X[tr], y[tr], g[tr], {"te": X[te]}, f"cv_{s}_{f}", s * 100 + f)
        Zte = Zte["te"]; Ptr, Pte = platt(Ztr, y[tr], Zte)
        for n in METODOS:
            p, yh, w = prever(n, Ztr, y[tr], Zte, Ptr, Pte)
            c = COMBINADORES[n][0] if n in COMBINADORES else [COLS.index(n)]
            custo = (t_int[c] + t_fin[c]).sum() if n in COMBINADORES else t_fin[c].sum()
            linhas.append(dict(metodo=n, tipo="combinação" if n in COMBINADORES else "modelo único", seed=s, fold=f,
                               n_tr=len(tr), n_te=len(te), treino_s=custo, **metricas(y[te], yh, p)))
            if w: pesos.append(dict(metodo=n, seed=s, fold=f, **w))
        msg = f"seed {s} fold {f}: {time.time() - t0:6.1f}s"
        print(msg, flush=True)
        with LOG.open("a") as fh: fh.write(msg + "\n")
R = pd.DataFrame(linhas); W = pd.DataFrame(pesos)
R.to_csv("stacking_cv.csv", sep=";", decimal=",", index=False)
''')

code(r'''
resumo = (R.groupby(["metodo", "tipo"])
           .agg(MCC=("mcc", "mean"), MCC_dp=("mcc", "std"), PR_AUC=("pr_auc", "mean"),
                real_lixeira_pct=("real_lixeira", lambda v: 100 * v.mean()), spam_passou_pct=("spam_passou", lambda v: 100 * v.mean()),
                treino_s=("treino_s", "mean"))
           .sort_values("MCC", ascending=False).reset_index(level="tipo"))
resumo.round({"MCC": 4, "MCC_dp": 4, "PR_AUC": 4, "real_lixeira_pct": 2, "spam_passou_pct": 2, "treino_s": 2})
''')

md("## 5. A diferença é real? Melhor método contra os rivais (15 folds)")
code(r'''
def t_corrigido(a, b, n_tr, n_te):
    """Nadeau & Bengio (2003): o t comum subestima a variancia porque os treinos dos folds se sobrepoem."""
    d = np.asarray(a) - np.asarray(b); J = len(d); v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)

P = R.pivot_table(index=["seed", "fold"], columns="metodo", values="mcc")
n_tr, n_te = R.n_tr.mean(), R.n_te.mean()
melhor = resumo.index[0]
melhor_comb = resumo[resumo.tipo == "combinação"].index[0]
melhor_unico = resumo[resumo.tipo == "modelo único"].index[0]
rivais = [m for m in dict.fromkeys(list(COMBINADORES) + list(resumo[resumo.tipo == "modelo único"].index[:3])) if m != melhor]
comp = pd.DataFrame([dict(contra=o, dif_MCC=d, t=t, p=p) for o in rivais
                     for d, t, p in [t_corrigido(P[melhor], P[o], n_tr, n_te)]]).sort_values("p").reset_index(drop=True)
comp["p_holm"] = np.minimum(1, np.maximum.accumulate((len(comp) - comp.index) * comp.p))
comp["veredito"] = np.where(comp.p_holm < 0.05, "melhor vence", "empate")
print("melhor (média de 15 folds):", melhor, "| melhor combinação:", melhor_comb, "| melhor modelo único:", melhor_unico)
comp.round(4)
''')
code(r'''
alvo = melhor_unico if melhor != melhor_unico else resumo.index[1]
pr = baycomp.two_on_single(P[melhor].values, P[alvo].values, rope=0.01, runs=len(REPETICOES))
print(f"Bayesiano correlacionado {melhor} x {alvo}: P({melhor} melhor) = {pr[0]:.3f} | "
      f"P(equivalentes, |dif MCC| < 0,01) = {pr[1]:.3f} | P({alvo} melhor) = {pr[2]:.3f}")
''')

md("## 6. Quem as combinações escolheram")
code(r'''
med = lambda nome: W[W.metodo == nome][COLS].mean()
L1 = W[W.metodo == "Stacking 13 + L1"][COLS]
pesos_tab = pd.DataFrame({
    "L1: % dos folds em que sobrevive": (L1.abs() > 1e-8).mean().mul(100),
    "L1: peso médio": L1.mean(),
    "Stacking 13: peso médio": med("Stacking 13 (todos)"),
    "Caruana: fatia média %": med("Caruana 13").mul(100),
}).sort_values("Caruana: fatia média %", ascending=False)
pesos_tab.round(2)
''')

md(r"""
## 7. Base de teste travada (20%): usada uma única vez
**Regra fixada antes de olhar:** finalistas = a melhor combinação na CV, o Stacking 4 (a referência anterior), o melhor modelo único na CV, o LinearSVC e a Regressão Logística. Todos são treinados nos 80% e avaliados nos 20%.
""")
code(r'''
FINALISTAS = list(dict.fromkeys([melhor_comb, "Stacking 4 (LR+NB+RF+LGBM)", melhor_unico, "LinearSVC", "Reg. Logística"]))
Ztr, Zt, _, _ = gerar_notas(X, y, g, {"teste": X_all[teste]}, "teste_travado", SEED)
Zt = Zt["teste"]; Ptr, Pt = platt(Ztr, y, Zt); yt = y_all[teste]
PRED = {n: prever(n, Ztr, y, Zt, Ptr, Pt)[:2] for n in FINALISTAS}

def mcc_np(a, b):
    tp = np.sum(a & b); tn = np.sum(~a & ~b); fp = np.sum(~a & b); fn = np.sum(a & ~b)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0

# bootstrap por grupo: sorteia textos distintos (copias identicas entram juntas)
_, inv = np.unique(g_all[teste], return_inverse=True)
membros = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
rng = np.random.default_rng(SEED); yb = yt.astype(bool)
boot = {n: [] for n in FINALISTAS}
for _ in range(2000):
    sel = np.concatenate([membros[k] for k in rng.integers(0, len(membros), len(membros))])
    for n in FINALISTAS:
        boot[n].append(mcc_np(yb[sel], PRED[n][1][sel].astype(bool)))
boot = pd.DataFrame(boot)

tab_teste = pd.DataFrame([dict(modelo=n, MCC_CV=resumo.loc[n, "MCC"], **metricas(yt, PRED[n][1], PRED[n][0]),
                               MCC_IC95=f"[{boot[n].quantile(.025):.3f}; {boot[n].quantile(.975):.3f}]",
                               reais_na_lixeira=int(((PRED[n][1] == 1) & (yt == 0)).sum()),
                               spams_que_passaram=int(((PRED[n][1] == 0) & (yt == 1)).sum())) for n in FINALISTAS]).set_index("modelo")
print(f"base de teste: {len(yt)} e-mails ({(yt == 0).sum()} reais, {(yt == 1).sum()} spam)")
tab_teste.round(4)
''')
code(r'''
for a, b in dict.fromkeys([(melhor_comb, melhor_unico), (melhor_comb, "LinearSVC")]):
    if a == b: continue
    ca, cb = PRED[a][1] == yt, PRED[b][1] == yt
    n_a, n_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_mc = stats.binomtest(min(n_a, n_b), n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    d = boot[a] - boot[b]
    print(f"{a} x {b}\n  McNemar: só {a} acerta = {n_a} | só {b} acerta = {n_b} | p = {p_mc:.3f}\n"
          f"  dif. MCC no teste = {tab_teste.loc[a, 'mcc'] - tab_teste.loc[b, 'mcc']:+.4f} | IC95 bootstrap "
          f"[{d.quantile(.025):+.4f}; {d.quantile(.975):+.4f}] | P(dif > 0) = {(d > 0).mean():.2f}")
''')

md("## 8. Validação externa: treina nos 5172 e-mails, testa no Enron2–6 (cópias do CSV removidas)")
code(r'''
EXT = next((p for p in [Path("../../dados/enron_vetores.npz"), Path("../dados_externos/enron_vetores.npz")] if p.exists()), Path("../../dados/enron_vetores.npz"))
if EXT.exists():
    z = np.load(EXT); Xext, yext = {}, {}
    for k in range(2, 7):
        A = z[f"X{k}"].astype(np.int64)
        novo = np.array([hashlib.md5(r.tobytes()).digest() not in hash_csv for r in A])
        Xext[f"Enron{k}"] = sparse.csr_matrix(A[novo].astype(np.float32)); yext[f"Enron{k}"] = z[f"y{k}"][novo]
        print(f"Enron{k}: {novo.sum():5d} e-mails ({(~novo).sum()} cópias do CSV removidas) | spam {yext[f'Enron{k}'].mean():.1%}")
    del A; gc.collect()
    Zf, Zext, _, _ = gerar_notas(X_all, y_all, g_all, Xext, "externo", SEED)
    lin, PE = [], {}
    for b in Xext:
        Pf, Pe = platt(Zf, y_all, Zext[b])
        for n in FINALISTAS:
            p, yh, _ = prever(n, Zf, y_all, Zext[b], Pf, Pe)
            PE[b, n] = yh
            lin.append(dict(base=b, modelo=n, **metricas(yext[b], yh, p)))
    E = pd.DataFrame(lin); E.to_csv("stacking_externo.csv", sep=";", decimal=",", index=False)
    for col, nome in [("mcc", "MCC"), ("pr_auc", "PR-AUC (não depende do limiar)"),
                      ("real_lixeira", "% de e-mails reais marcados como spam"), ("spam_passou", "% de spam que passou")]:
        t = E.pivot(index="modelo", columns="base", values=col).reindex(FINALISTAS)
        t["média"] = t.mean(1); pct = col in ("real_lixeira", "spam_passou")
        print(f"\n== {nome} ==\n" + (t * 100 if pct else t).round(1 if pct else 3).to_string())
    print(f"\n== McNemar por base: {melhor_comb} x LinearSVC ==")
    for b in Xext:
        ca, cb = PE[b, melhor_comb] == yext[b], PE[b, "LinearSVC"] == yext[b]
        n_a, n_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mc = stats.binomtest(min(n_a, n_b), n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
        print(f"{b}: só a combinação acerta = {n_a:4d} | só o LinearSVC acerta = {n_b:4d} | p = {p_mc:.2g}")
else:
    E = None; print("enron_vetores.npz não encontrado: validação externa pulada")
''')

md("## 9. Visão única dos finalistas")
code(r'''
visao = pd.DataFrame({
    "MCC CV (80%)": resumo.loc[FINALISTAS, "MCC"],
    "MCC teste travado (20%)": tab_teste["mcc"],
    "MCC externo (média Enron2–6)": E.groupby("modelo").mcc.mean() if E is not None else np.nan,
    "% reais na lixeira (externo)": E.groupby("modelo").real_lixeira.mean() * 100 if E is not None else np.nan,
    "treino (s)": resumo.loc[FINALISTAS, "treino_s"],
}).reindex(FINALISTAS)
visao.round(3)
''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1]); print("ok", sys.argv[1])
