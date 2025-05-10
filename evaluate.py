#!/usr/bin/env python
"""Evaluate k-NN quality (Top-5 majority-vote accuracy)."""
import pickle, numpy as np, pandas as pd, faiss, pathlib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

BIN   = pathlib.Path("faiss_index.bin")
PICKL = pathlib.Path("id_lookup.pkl")
CSV   = pathlib.Path("clean_features.csv")
assert BIN.exists() and PICKL.exists() and CSV.exists(), "Run build_index.py first"

# load artifacts
idx   = faiss.read_index(str(BIN))
ids   = pickle.load(open(PICKL, "rb"))
df    = pd.read_csv(CSV)
Xall  = df.drop(columns=["id"]).values.astype("float32")

# labels: in this file column name is 'Diabetes_binary'
y = df["Diabetes_binary"].values

# train/val split on row indices
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

# fit scaler on train
scaler = StandardScaler().fit(Xall[train_idx])
def embed(v):
    v = scaler.transform(v).astype("float32")
    v = np.ascontiguousarray(v)
    faiss.normalize_L2(v)
    return v

hits = 0
for i in test_idx:
    q   = embed([Xall[i]])
    _, I = idx.search(q, 5)
    nbr_ids = ids[I[0]]
    nbr_labels = y[nbr_ids]
    maj = Counter(nbr_labels).most_common(1)[0][0]
    hits += int(maj == y[i])

acc = hits / len(test_idx)
print(f"🏁  Top-5 majority-vote accuracy: {acc:.3%}  on {len(test_idx)} hold-out samples")

# save as JSON so the metric can be read programmatically
json.dump({"top5_majority_accuracy": acc}, open("eval_m2.json", "w"))
print("📄  eval_m2.json written")
