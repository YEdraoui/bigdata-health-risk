#!/usr/bin/env python
"""
Builds FAISS cosine-similarity index for the diabetes dataset.

Usage:
    python build_index.py --in data/diabetes.csv --k 5
"""
import argparse, pickle, pathlib
import numpy as np, pandas as pd, faiss
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--in",  dest="csv", default="data/diabetes.csv")
p.add_argument("--k",   dest="k",   type=int, default=5, help="Neighbours to keep")
args = p.parse_args()

csv_path = pathlib.Path(args.csv)
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found")

print("▶ loading CSV …")
df  = pd.read_csv(csv_path).dropna()
df.insert(0, "id", np.arange(len(df)))

print("▶ preprocessing …")
ids = df["id"].values.astype(np.int64)
X   = df.drop(columns=["id"]).values.astype("float32")
X   = StandardScaler().fit_transform(X)
X   = np.ascontiguousarray(X)              # for faiss
faiss.normalize_L2(X)

print("▶ building index …")
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

faiss.write_index(index, "faiss_index.bin")
pickle.dump(ids, open("id_lookup.pkl", "wb"))
df.to_csv("clean_features.csv", index=False)

print(f"✅  index ready → {index.ntotal} vectors  |  files written:")
print("   • clean_features.csv\n   • faiss_index.bin\n   • id_lookup.pkl")
