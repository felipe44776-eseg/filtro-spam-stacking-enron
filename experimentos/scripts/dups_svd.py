import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
df = pd.read_csv(sys.argv[1]); y = df.pop("Prediction").values; ids = df.pop("Email No."); X = df
key = pd.util.hash_pandas_object(X, index=False).values
g = pd.DataFrame({"key": key, "y": y, "tot": X.sum(1).values, "nz": (X > 0).sum(1).values, "id": ids.str.replace("Email ", "").astype(int)})
sz = g.groupby("key")["y"].transform("size"); g["copias"] = sz
print("== DUPLICATAS ==")
print("linhas em grupos com >1 copia:", int((sz > 1).sum()), "| grupos:", g[sz > 1]["key"].nunique(), "| linhas excedentes:", int(X.duplicated().sum()))
t = g.assign(dup=sz > 1).groupby("dup")["y"].agg(["size", "mean"]); t["mean"] = (t["mean"]*100).round(1); print("spam% por (tem copia?):\n", t)
ex = g[X.duplicated().values]
print("linhas excedentes por classe:", ex["y"].value_counts().to_dict(), "| taxa de excedente: ham", round((ex.y==0).sum()/(y==0).sum()*100,1), "% spam", round((ex.y==1).sum()/(y==1).sum()*100,1), "%")
print("tamanho do grupo (copias) dist:", g[sz>1].drop_duplicates("key")["copias"].value_counts().sort_index().to_dict())
print("maior grupo spam / ham:", g[(sz>1)&(g.y==1)].copias.max(), g[(sz>1)&(g.y==0)].copias.max())
print("palavras distintas (mediana) dup vs unico:", g[sz>1].nz.median(), g[sz==1].nz.median())
print("total palavras (mediana) dup vs unico:", g[sz>1].tot.median(), g[sz==1].tot.median())
# spread of ids inside groups (copies sent at different times?)
spread = g[sz>1].groupby("key")["id"].agg(lambda s: s.max()-s.min())
print("distancia entre Email No. dentro do grupo (mediana/max):", spread.median(), spread.max())
# top words in largest duplicate groups
big = g[sz>1].drop_duplicates("key").sort_values("copias", ascending=False).head(6)
for _, r in big.iterrows():
    row = X[key == r.key].iloc[0]; top = row[row > 0].sort_values(ascending=False)
    print(f"  grupo copias={r.copias} y={r.y} nz={r.nz}: ", list(top.index[:12]))
# accuracy of 'has copy -> spam' rule
print("regra 'tem copia => spam' acuracia:", round(((sz>1).astype(int).values == y).mean(),3))
print("\n== SVD (TF-IDF, sem duplicatas) ==")
Xd = X[~X.duplicated()]; yd = y[~X.duplicated().values]
T = TfidfTransformer(sublinear_tf=True).fit_transform(Xd.values)
svd = TruncatedSVD(100, random_state=42).fit(T)
ev = svd.explained_variance_ratio_
print("variancia explicada: 2 comp", round(ev[:2].sum()*100,1), "% | 10", round(ev[:10].sum()*100,1), "% | 100", round(ev.sum()*100,1), "%")
Z = svd.transform(T); cols = np.array(Xd.columns)
for i in range(4):
    c = svd.components_[i]; o = np.argsort(c)
    corr = np.corrcoef(Z[:, i], yd)[0,1]
    print(f"  comp{i+1} corr c/ spam={corr:+.2f} | +: {list(cols[o[::-1][:8]])} | -: {list(cols[o[:8]])}")
