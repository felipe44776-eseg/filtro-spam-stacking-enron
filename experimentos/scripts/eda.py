import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1])
print("shape", df.shape)
print("first/last cols", list(df.columns[:3]), list(df.columns[-3:]))
print("dtypes", df.dtypes.value_counts().to_dict())
print("nulls total", int(df.isna().sum().sum()))
y = df["Prediction"]; X = df.drop(columns=["Email No.", "Prediction"])
print("class balance", y.value_counts().to_dict(), "spam%", round(y.mean()*100,2))
print("Email No. unique", df["Email No."].nunique(), df["Email No."].head(3).tolist())
# duplicate column names mangled?
mangled = [c for c in X.columns if "." in c]
print("mangled/dotted col names", mangled[:20], len(mangled))
Xv = X.values
print("sparsity % zeros", round((Xv==0).mean()*100,2))
print("max count", Xv.max(), "mean nonzero per row", round((Xv>0).sum(1).mean(),1))
rowsum = Xv.sum(1)
print("emails with all-zero features", int((rowsum==0).sum()))
print("total words/email by class:\n", pd.Series(rowsum).groupby(y.values).describe().round(1))
dup_all = X.duplicated()
print("duplicate feature rows", int(dup_all.sum()))
g = pd.DataFrame(Xv).assign(y=y.values)
conf = g.groupby(list(range(Xv.shape[1])))["y"].nunique()
print("conflicting duplicate groups (same X, diff y)", int((conf>1).sum()))
zero_cols = (Xv.sum(0)==0).sum(); print("all-zero columns", int(zero_cols))
df_cnt = (Xv>0).sum(0)
print("doc freq: cols in <5 docs", int((df_cnt<5).sum()), "| <1% docs", int((df_cnt<0.01*len(X)).sum()))
# discriminative words: doc freq ratio
p_spam = (X[y==1]>0).mean(); p_ham = (X[y==0]>0).mean()
eps=1e-3
lr = np.log((p_spam+eps)/(p_ham+eps))
print("top spam words:", lr.sort_values(ascending=False).head(25).round(2).to_dict())
print("top ham words:", lr.sort_values().head(25).round(2).to_dict())
for w in ["enron","ect","hou","vince","kaminski","daren","hpl","meter","forwarded","subject"]:
    if w in X: print(f"  {w}: ham_df={p_ham[w]:.3f} spam_df={p_spam[w]:.3f}")
# single-feature leak check: accuracy of rule "contains enron/ect/hou -> ham"
rule = ((X[["enron","ect","hou"]]>0).any(axis=1)).astype(int)
pred = 1-rule
print("rule 'has enron|ect|hou => ham' accuracy", round((pred==y).mean(),4))
