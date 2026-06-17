import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def build_dynamic_dataset(log_dir):
    all_logs_data = []
    all_seen_syscalls = set()
    
    print("Parsing logs and dynamically discovering system call IDs...")
    
    for root, dirs, files in os.walk(log_dir):
        for filename in files:
            if filename.endswith(".txt"):
                path = os.path.join(root, filename)
                
                rows = []
                with open(path) as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            rows.append(parts[2].strip())

                bigrams = [f"{rows[i]}-{rows[i+1]}" for i in range(len(rows)-1)]
                counts = Counter(bigrams)
                all_seen_syscalls.update(counts.keys())
                
                label = "attack" if "attack" in root.lower() or filename.startswith("attack_") else "normal"
                
                all_logs_data.append({
                    "source": os.path.relpath(path, log_dir),
                    "counts": counts,
                    "total": max(len(bigrams), 1),
                    "actual_class": label
                })
    
    syscall_list = sorted(list(all_seen_syscalls))
    vectors = []
    
    for item in all_logs_data:
        counts = item["counts"]
        total = item["total"]
        
        features = {s: (counts.get(s, 0) / total) for s in syscall_list}
        features["source"] = item["source"]
        features["actual_class"] = item["actual_class"]
        vectors.append(features)
        
    return pd.DataFrame(vectors), syscall_list

# --- Execution ---

df, dynamic_syscalls = build_dynamic_dataset("logs")
print(f"Dataset built! Found {len(dynamic_syscalls)} unique bigram columns across {len(df)} files.\n")

# Prepare features (X) and target labels (y)
X = df[dynamic_syscalls]
y = df["actual_class"]

# Split data: 70% to train the model, 30% to hide away and test its true accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training Supervised Random Forest Classifier on {len(X_train)} files...")
# Initialize a Supervised Classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

print("Evaluating model on the unseen test pool (30%)...")
y_pred = clf.predict(X_test)

print("\n" + "="*40)
print("     SUPERVISED CONFUSION MATRIX")
print("="*40)
# Create the matrix from the evaluation slice
matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(matrix)
print("="*40)

print("\nDetailed Performance Metrics:")
print(classification_report(y_test, y_pred))