import sys, os, re, glob, random, numpy as np, pandas as pd
from collections import Counter
csv, root = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv); y = df.pop("Prediction").values; df.pop("Email No."); vocab = list(df.columns)
csv_keys = Counter(map(bytes, df.values.astype(np.int32)))
files = sorted(glob.glob(os.path.join(root, "enron1", "*", "*.txt")))
random.seed(0); sample = random.sample(files, 300)
read = lambda f: open(f, encoding="latin-1").read()
vidx = {w: i for i, w in enumerate(vocab)}
def h_substr(t):        return [t.count(w) for w in vocab]
def h_substr_lower(t):  t = t.lower(); return [t.count(w) for w in vocab]
def h_tokens(t):
    c = Counter(re.findall(r"[a-zA-Z]+", t.lower())); return [c.get(w, 0) for w in vocab]
def h_substr_overlap(t):
    return [len(re.findall("(?=" + re.escape(w) + ")", t)) for w in vocab]
for name, fn in [("substring case-sensitive", h_substr), ("substring lower", h_substr_lower), ("tokens [a-z]+", h_tokens)]:
    hit = sum(bytes(np.array(fn(read(f)), dtype=np.int32)) in csv_keys for f in sample)
    print(f"{name:28s} match {hit}/300")
# diagnose closest: for a sample file, find nearest CSV row under best hypothesis
