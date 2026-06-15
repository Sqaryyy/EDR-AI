import os
import pandas as pd
from collections import Counter
from sklearn.ensemble import IsolationForest
import json

SYSCALLS = ["read", "write", "open", "close", "stat", "mmap", 
            "munmap", "socket", "connect", "clone", "fork", 
            "execve", "rename", "openat", "openat2", "renameat2", "other"]

def extract_features(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                rows.append(parts[2])

    counts = Counter(rows)
    features = {s: counts.get(s, 0) for s in SYSCALLS}
    features["total"] = len(rows)
    
    # derived features - these catch behavioral patterns
    total = max(len(rows), 1)
    features["rename_ratio"] = (counts.get("rename", 0) + counts.get("renameat2", 0)) / total
    features["write_ratio"] = counts.get("write", 0) / total
    features["openat_ratio"] = counts.get("openat", 0) / total

    return features

def build_dataset(log_dir):
    vectors = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            path = os.path.join(log_dir, filename)
            features = extract_features(path)
            features["source"] = filename
            vectors.append(features)
    return pd.DataFrame(vectors)

# build dataset from logs
df = build_dataset("logs")
print("Dataset:")
print(df)

# train isolation forest on numeric columns only
feature_cols = SYSCALLS + ["total", "rename_ratio", "write_ratio", "openat_ratio"]
X = df[feature_cols]

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# score each log file
df["anomaly_score"] = model.decision_function(X)
df["is_anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal

print("\nScores:")
print(df[["source", "anomaly_score", "is_anomaly"]])