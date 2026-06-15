import os
import numpy as np
from collections import defaultdict

SYSCALLS = ["read", "write", "open", "close", "stat", "mmap",
            "munmap", "socket", "connect", "clone", "fork",
            "execve", "rename", "openat", "openat2", "renameat2", "other"]

# map syscall name to integer index
SYSCALL_TO_IDX = {s: i for i, s in enumerate(SYSCALLS)}
VOCAB_SIZE = len(SYSCALLS)

def log_to_sequence(filepath):
    seq = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                name = parts[2].strip()
                idx = SYSCALL_TO_IDX.get(name, SYSCALL_TO_IDX["other"])
                seq.append(idx)
    return seq

def sequence_to_ngrams(seq, n=5):
    ngrams = []
    for i in range(len(seq) - n):
        x = seq[i:i+n]       # input window
        y = seq[i+n]          # next syscall to predict
        ngrams.append((x, y))
    return ngrams

def load_normal_logs(log_dir, n=5):
    X, Y = [], []
    for filename in os.listdir(log_dir):
        if filename.startswith("normal_") and filename.endswith(".txt"):
            path = os.path.join(log_dir, filename)
            seq = log_to_sequence(path)
            for x, y in sequence_to_ngrams(seq, n):
                X.append(x)
                Y.append(y)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = load_normal_logs("logs")
    print(f"Total training samples: {len(X)}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Sample input: {X[0]} -> predict: {Y[0]}")
    print(f"Decoded: {[SYSCALLS[i] for i in X[0]]} -> {SYSCALLS[Y[0]]}")