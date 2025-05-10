from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, faiss, pickle, pandas as pd
from sklearn.preprocessing import StandardScaler

# one-time load
index = faiss.read_index("faiss_index.bin")
ids   = pickle.load(open("id_lookup.pkl", "rb"))
df    = pd.read_csv("clean_features.csv")
scaler = StandardScaler().fit(df.drop(columns=["id"]))

app = FastAPI(title="Diabetes k-NN API")

class Query(BaseModel):
    vector: list[float]
    k: int = 5

@app.post("/nearest")
def nearest(q: Query):
    v = scaler.transform([q.vector]).astype("float32")
    v = np.ascontiguousarray(v)
    faiss.normalize_L2(v)
    D, I = index.search(v, q.k)
    return {"ids": ids[I[0]].tolist(), "scores": D[0].tolist()}

# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Health-check endpoint."""
    return {"status": "ok", "docs": "/docs"}
# ---------------------------------------------------------------------------
