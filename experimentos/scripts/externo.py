import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from collections import Counter
from multiprocessing import Pool
VOCAB = None
def init(v):
    global VOCAB; VOCAB = v
def vec(f):
    t = open(f, encoding="latin-1").read(); return [t.count(w) for w in VOCAB]
def main():
    csv, root, cache = sys.argv[1], sys.argv[2], sys.argv[3]
    df = pd.read_csv(csv); y1 = df.pop("Prediction").values; df.pop("Email No."); vocab = list(df.columns); X1 = df.values
    if not os.path.exists(cache):
        out = {}
        with Pool(8, initializer=init, initargs=(vocab,)) as p:
            for n in range(1, 7):
                fs = sorted(glob.glob(os.path.join(root, f"enron{n}", "*", "*.txt")))
                out[f"X{n}"] = np.array(p.map(vec, fs, chunksize=50), dtype=np.int32)
                out[f"y{n}"] = np.array([1 if os.sep + "spam" + os.sep in f or "/spam/" in f else 0 for f in fs])
                print("vetorizado enron", n, out[f"X{n}"].shape, flush=True)
        np.savez_compressed(cache, **out)
    Z = np.load(cache)
    # 1. reproducao integral do Enron1
    a = Counter(map(bytes, X1.astype(np.int32))); b = Counter(map(bytes, Z["X1"]))
    print("Enron1 reproduzido == CSV (multiconjunto):", a == b, "| rotulos batem:", sorted(y1) == sorted(Z["y1"]))
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, accuracy_score
    models = {
        "MNB": lambda: MultinomialNB(),
        "CNB": lambda: ComplementNB(),
        "TFIDF+LR": lambda: make_pipeline(TfidfTransformer(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
        "TFIDF+LinSVC": lambda: make_pipeline(TfidfTransformer(), LinearSVC(C=1, class_weight="balanced")),
        "RF": lambda: RandomForestClassifier(400, n_jobs=-1, random_state=42, class_weight="balanced"),
        "LGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=42),
        "kNN tfidf cos": lambda: make_pipeline(TfidfTransformer(), KNeighborsClassifier(5, metric="cosine")),
    }
    leak = ["enron","daren","farmer","hpl","sitara","mmbtu","ect","hou","vance","melissa","cotten","chokshi","tenaska","forwarded","paliourg","kaminski","vince","clynes","pat","ami","gary","robert","smith","george","mary","noms","nom","actuals","xls"]
    keep_dom = np.array([w not in leak for w in vocab])
    keep_short = np.array([len(w) >= 3 for w in vocab])
    variants = {"todas 3000": np.ones(len(vocab), bool), "sem dominio": keep_dom, "sem dominio e <3 letras": keep_dom & keep_short}
    train_keys = set(map(bytes, X1.astype(np.int32)))
    rows = []
    for vname, mask in variants.items():
        for mname, mk in models.items():
            if vname != "todas 3000" and mname not in ("TFIDF+LR", "LGBM", "MNB"): continue
            m = mk().fit(X1[:, mask], y1)
            for n in range(2, 7):
                X, y = Z[f"X{n}"], Z[f"y{n}"]
                novo = np.array([bytes(r) not in train_keys for r in X])
                X, y = X[novo][:, mask], y[novo]
                p = m.predict(X)
                s = m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
                rows.append(dict(variante=vname, modelo=mname, base=f"enron{n}", n=len(y), excl_copias=int((~novo).sum()), spam_pct=round(y.mean()*100,1),
                    acc=accuracy_score(y,p), prec=precision_score(y,p,zero_division=0), rec=recall_score(y,p), f1=f1_score(y,p), mcc=matthews_corrcoef(y,p),
                    auc=roc_auc_score(y,s), fpr_ham=((p==1)&(y==0)).sum()/(y==0).sum(), fnr_spam=((p==0)&(y==1)).sum()/(y==1).sum()))
    R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    R.to_csv(cache.replace(".npz", "_resultados.csv"), sep=";", index=False, decimal=",")
    print(R[["base","n","excl_copias","spam_pct"]].drop_duplicates("base").to_string(index=False))
    for v in variants:
        print(f"\n=== {v} ===")
        piv = R[R.variante==v].pivot_table(index="modelo", columns="base", values=["mcc","auc","fpr_ham","fnr_spam"]).round(3)
        print(piv.to_string())
if __name__ == "__main__":
    main()
