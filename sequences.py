import os
import numpy as np
import json

def build_vocab(log_dirs):
    syscalls = set()
    for log_dir in log_dirs:
        for root, dirs, files in os.walk(log_dir):
            for filename in files:
                if filename.endswith(".txt"):
                    with open(os.path.join(root, filename)) as f:
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) == 3:
                                syscalls.add(parts[2].strip())
    vocab = sorted(syscalls)
    return {s: i for i, s in enumerate(vocab)}, vocab

def load_vocab():
    with open("model/vocab.json") as f:
        data = json.load(f)
    return data["syscall_to_idx"], data["syscalls"]

def build_and_save_vocab():
    os.makedirs("model", exist_ok=True)
    syscall_to_idx, syscalls = build_vocab(["logs"])
    with open("model/vocab.json", "w") as f:
        json.dump({"syscall_to_idx": syscall_to_idx, "syscalls": syscalls}, f)
    return syscall_to_idx, syscalls

def log_to_sequence(filepath, syscall_to_idx):
    seq = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                name = parts[2].strip()
                idx = syscall_to_idx.get(name, 0)
                seq.append(idx)
    return seq

def sequence_to_ngrams(seq, n=5):
    ngrams = []
    for i in range(len(seq) - n):
        x = seq[i:i+n]
        y = seq[i+n]
        ngrams.append((x, y))
    return ngrams

def load_normal_logs(log_dir, syscall_to_idx, n=5):
    X, Y = [], []

    for filename in os.listdir(log_dir):
        if filename.startswith("normal_") and filename.endswith(".txt"):
            path = os.path.join(log_dir, filename)
            seq = log_to_sequence(path, syscall_to_idx)
            for x, y in sequence_to_ngrams(seq, n):
                X.append(x)
                Y.append(y)

    adfa_normal = os.path.join(log_dir, "adfa_normal")
    if os.path.exists(adfa_normal):
        for filename in os.listdir(adfa_normal):
            if filename.endswith(".txt"):
                path = os.path.join(adfa_normal, filename)
                seq = log_to_sequence(path, syscall_to_idx)
                for x, y in sequence_to_ngrams(seq, n):
                    X.append(x)
                    Y.append(y)

    return np.array(X), np.array(Y)

if __name__ == "__main__":
    syscall_to_idx, syscalls = build_and_save_vocab()
    X, Y = load_normal_logs("logs", syscall_to_idx)
    print(f"Total training samples: {len(X)}")
    print(f"Vocab size: {len(syscalls)}")
    print(f"Syscalls seen: {syscalls}")