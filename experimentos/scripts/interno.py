import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
df = pd.read_csv(sys.argv[1]); y = df.pop("Prediction").values; df.pop("Email No."); X = df.values
grp = pd.util.hash_pandas_object(df, index=False).values
ms = {"MNB": MultinomialNB(), "TFIDF+LR": make_pipeline(TfidfTransformer(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
      "TFIDF+LinSVC": make_pipeline(TfidfTransformer(), LinearSVC(C=1, class_weight="balanced")),
      "RF": RandomForestClassifier(400, n_jobs=-1, random_state=42, class_weight="balanced"),
      "LGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=42)}
for split, cv, g in [("ingenuo (KFold)", StratifiedKFold(5, shuffle=True, random_state=42), None), ("por grupo (copias juntas)", StratifiedGroupKFold(5, shuffle=True, random_state=42), grp)]:
    print(f"--- {split}")
    for n, m in ms.items():
        p = cross_val_predict(m, X, y, cv=cv, groups=g)
        print(f"  {n:13s} MCC={matthews_corrcoef(y,p):.3f} FPR_ham={((p==1)&(y==0)).sum()/(y==0).sum():.3f} FNR_spam={((p==0)&(y==1)).sum()/(y==1).sum():.3f}")
