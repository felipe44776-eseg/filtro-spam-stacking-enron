import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from lightgbm import LGBMClassifier
df = pd.read_csv(sys.argv[1]); y = df.pop("Prediction").values; df.pop("Email No."); X = df
# substring hypothesis
print("ect min/median:", X["ect"].min(), X["ect"].median())
print("single letters mean/email:", {c: round(X[c].mean(),1) for c in ["a","e","s","t","d"] if c in X})
print("rows the>=there+these+them+they:", round((X["the"] >= X[["there","these","them","they"]].sum(1)).mean(),3))
print("rows hou>=house(if any):", "house" in X)
print("rows 'a' count vs total words ratio median:", round((X["a"]/X.sum(1)).median(),3))
# dedup
Xd = X[~X.duplicated()]; yd = y[~X.duplicated().values]
print("dedup shape", Xd.shape, "spam%", round(yd.mean()*100,1))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
sc = ["f1","precision","recall","roc_auc","average_precision","matthews_corrcoef"]
log1p = FunctionTransformer(np.log1p)
models = {
 "MNB": MultinomialNB(),
 "CNB": ComplementNB(),
 "BNB": BernoulliNB(binarize=0.5),
 "TFIDF+LR": make_pipeline(TfidfTransformer(), LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
 "TFIDF+LinSVC": make_pipeline(TfidfTransformer(), LinearSVC(C=1, class_weight="balanced")),
 "RF": RandomForestClassifier(400, n_jobs=-1, random_state=42, class_weight="balanced"),
 "LGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=42),
 "kNN raw(k=5)": KNeighborsClassifier(5),
 "kNN tfidf cos(k=5)": make_pipeline(TfidfTransformer(), KNeighborsClassifier(5, metric="cosine")),
}
def run(Xm, label):
    print(f"\n--- {label} | {Xm.shape}")
    for n, m in models.items():
        r = cross_validate(m, Xm.values, yd, cv=cv, scoring=sc, n_jobs=1)
        print(f"{n:20s} " + " ".join(f"{s[:5]}={r['test_'+s].mean():.3f}" for s in sc))
run(Xd, "dedup, all 3000 words")
leak = ["enron","daren","farmer","hpl","sitara","mmbtu","ect","hou","vance","melissa","cotten","chokshi","tenaska","forwarded","paliourg","kaminski","vince","clynes","pat","ami","gary","robert","smith","george","mary","noms","nom","actuals","xls"]
leak = [w for w in leak if w in Xd]
run(Xd.drop(columns=leak), f"dedup, sem {len(leak)} palavras de dominio/nomes")
